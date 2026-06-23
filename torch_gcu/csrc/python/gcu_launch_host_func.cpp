// #include <cuda_runtime.h>
#include "python/gcu_launch_host_func.h"

#include <Python.h>
#include <tops/tops_runtime_api.h>
#include <torch/csrc/utils/pybind.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

#include "gcu/gcu_graph.h"
#include "gcu/gcu_stream.h"

// topsError_t topsLaunchHostFunc(topsStream_t stream, topsHostFn_t fn, void*
// userData);
namespace py = pybind11;

namespace torch_gcu {
#ifdef RUNTIME_1_5_1_1016
// 带参数的主机回调函数结构体
struct CallbackData {
 public:
  CallbackData(py::function py_func, py::tuple args, py::dict kwargs)
      : py_func(py_func),
        args(args),
        kwargs(kwargs),
        is_valid(true),
        completed(false),
        cached(false) {}  // 初始化cached标志

  ~CallbackData() {
    // 确保在析构时标记为无效
    // is_valid = false;
  }

 public:
  py::function py_func;
  py::tuple args;
  py::dict kwargs;
  std::atomic<bool> is_valid;  // 原子标志，确保线程安全

  // 同步相关成员
  std::atomic<bool> completed;  // 标记回调是否完成
  std::atomic<bool> cached;  // 标记是否已缓存（防止graph replay时重复缓存）
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
};

// Python回调工作线程系统
class PythonCallbackWorker {
 private:
  std::thread worker_thread;
  std::queue<CallbackData*> callback_queue;
  std::queue<CallbackData*>
      cached_callback_queue;  // 保存用于graph replay的回调数据
  std::mutex queue_mutex;
  std::mutex cached_queue_mutex;  // 保护cached队列的独立锁
  std::condition_variable cv;
  bool should_stop = false;

  void workerLoop() {
    while (true) {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv.wait(lock, [this] { return !callback_queue.empty() || should_stop; });
      if (should_stop && callback_queue.empty()) {
        break;
      }

      if (!callback_queue.empty()) {
        CallbackData* data = callback_queue.front();
        callback_queue.pop();
        lock.unlock();
        // 在工作线程中执行Python回调
        executePythonCallback(data);

        // 标记回调完成并通知等待的线程
        {
          std::lock_guard<std::mutex> completion_lock(data->completion_mutex);
          data->completed.store(true);
        }
        data->completion_cv.notify_all();

        // 注意：在同步模式下，不在这里删除数据
        // 数据将在enqueueCallback返回后由调用者负责删除
      }
    }
  }

  void executePythonCallback(CallbackData* data) {
    // 检查Python解释器是否可用
    if (!Py_IsInitialized() || Py_IsFinalizing()) {
      // Python正在finalizing，跳过callback执行
      return;
    }

    PyGILState_STATE gil_state;
    bool gil_acquired = false;

    // ❌ 不要使用 Py_BEGIN_ALLOW_THREADS！
    // 工作线程初始时没有GIL，使用该宏会导致 segmentation fault
    // 正确做法：直接使用 PyGILState_Ensure/Release
    try {
      // 在工作线程中获取GIL（工作线程初始时没有GIL）
      gil_state = PyGILState_Ensure();
      gil_acquired = true;

      try {
        // 确保Python对象仍然有效
        if (data && data->is_valid.load() && data->py_func) {
          if (!data->kwargs.empty()) {
            data->py_func(*data->args, **data->kwargs);
          } else if (data->args.size() > 0) {
            data->py_func(*data->args);
          } else {
            data->py_func();
          }
        }
      } catch (py::error_already_set& e) {
        printf("Python callback error: %s\n", e.what());
        e.restore();
        PyErr_Print();
        PyErr_Clear();
      } catch (const std::exception& e) {
        printf("Python callback C++ error: %s\n", e.what());
      }

    } catch (const std::exception& e) {
      printf("Worker thread error: %s\n", e.what());
    } catch (...) {
      printf("Unknown worker thread error\n");
    }

    // 确保GIL被正确释放
    if (gil_acquired) {
      try {
        PyGILState_Release(gil_state);
      } catch (...) {
        printf("Error releasing GIL in worker thread\n");
      }
    }
  }

  void deletePythonData(CallbackData* data) {
    if (!data) return;

    // 检查Python解释器是否正在finalizing
    // 如果是，跳过删除，让Python自己清理
    if (Py_IsInitialized() && !Py_IsFinalizing()) {
      // ❌ 不要使用 Py_BEGIN_ALLOW_THREADS！
      // 这个函数可能从工作线程调用，使用该宏会崩溃
      try {
        // 获取GIL来安全删除Python对象
        PyGILState_STATE gil_state = PyGILState_Ensure();

        try {
          // 在GIL保护下删除数据
          // pybind11对象的析构函数会在这里被调用
          delete data;
        } catch (...) {
          // Exception during Python object destruction
        }

        PyGILState_Release(gil_state);

      } catch (...) {
        // 如果无法获取GIL，静默失败
        // 在finalizing阶段，Python会自己清理
      }
    }
    // 如果Python正在finalizing，不做任何操作，避免GIL错误
  }

 private:
  static std::unique_ptr<PythonCallbackWorker> instance;
  static std::mutex instance_mutex;

 public:
  static PythonCallbackWorker& getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (!instance) {
      instance = std::make_unique<PythonCallbackWorker>();
    }
    return *instance;
  }

  static void destroyInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (instance) {
      instance->stop();
      instance.reset();
    }
  }

  PythonCallbackWorker() {
    worker_thread = std::thread(&PythonCallbackWorker::workerLoop, this);
    // 给工作线程一点时间启动
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ~PythonCallbackWorker() { stop(); }

  void enqueueCallback(CallbackData* data) {
    // 重置completed标志以支持graph replay
    // 必须在加入队列之前重置，并使用completion_mutex保护
    {
      std::lock_guard<std::mutex> completion_lock(data->completion_mutex);
      data->completed.store(false);
    }

    // 将回调加入队列
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      callback_queue.push(data);
    }

    // 通知工作线程
    cv.notify_one();

    // 同步等待回调完成
    std::unique_lock<std::mutex> completion_lock(data->completion_mutex);
    data->completion_cv.wait(completion_lock,
                             [data] { return data->completed.load(); });
    completion_lock.unlock();

    // 将数据缓存以支持graph replay（不删除，需要保持存活）
    // 使用原子操作防止graph replay时重复缓存同一个指针
    bool expected = false;
    if (data->cached.compare_exchange_strong(expected, true)) {
      // 只有第一次callback时才缓存，避免replay时重复添加导致double-free
      std::lock_guard<std::mutex> cached_lock(cached_queue_mutex);
      cached_callback_queue.push(data);
    }
  }

  void stop() {
    if (worker_thread.joinable()) {
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        should_stop = true;

        // 标记所有待处理的回调数据为无效
        std::queue<CallbackData*> temp_queue = callback_queue;
        while (!temp_queue.empty()) {
          CallbackData* data = temp_queue.front();
          temp_queue.pop();
          if (data) {
            data->is_valid.store(false);
          }
        }
      }
      cv.notify_all();

      // 等待工作线程完成
      try {
        if (worker_thread.joinable()) {
          worker_thread.join();
        }
      } catch (...) {
        // Exception while joining worker thread
      }

      // 清理剩余的回调数据
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        while (!callback_queue.empty()) {
          CallbackData* data = callback_queue.front();
          callback_queue.pop();
          if (data) {
            data->is_valid.store(false);
            deletePythonData(data);
          }
        }
      }

      // 清理缓存的回调数据（用于graph replay）
      {
        std::lock_guard<std::mutex> cached_lock(cached_queue_mutex);
        while (!cached_callback_queue.empty()) {
          CallbackData* data =
              cached_callback_queue.front();  // 修复：使用正确的队列
          cached_callback_queue.pop();
          if (data) {
            data->is_valid.store(false);
            deletePythonData(data);
          }
        }
      }
    }
  }
};

// tops回调包装器 - 正确处理GIL避免死锁
void hostCallbackWrapper(void* userData) {
  auto* data = static_cast<CallbackData*>(userData);

  // 同步模式：等待回调执行完成
  PythonCallbackWorker::getInstance().enqueueCallback(data);
}
#endif

// 封装topsLaunchHostFunc的接口
void launchHostFunc(py::function py_func, py::tuple args = py::tuple(),
                    py::dict kwargs = py::dict()) {
#ifdef RUNTIME_1_5_1_1016
  // 创建持久化回调数据
  if (current_gcu_graph_ != nullptr) {
    current_gcu_graph_->has_py_host_functions_ = true;
  }
  auto* data = new CallbackData(py_func, args, kwargs);
  auto stream = getCurrentGCUStream();
  // 调用tops API
  topsError_t err = topsLaunchHostFunc(stream, &hostCallbackWrapper, data);
  if (err != topsSuccess) {
    delete data;
    throw std::runtime_error(topsGetErrorString(err));
  }
#else
  std::cout << "launchHostFunc is not supported in this runtime version."
            << std::endl;
#endif
}

// Python绑定注册函数
// 注意：这个函数会在InitGcuBindings.cpp中被调用来注册Python接口
void RegisterLaunchHostFunc(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("launch_host_func", &launchHostFunc, py::arg("func"),
        py::arg("args") = py::tuple(), py::arg("kwargs") = py::dict(),
        "Launch a parameterized TOPS host function using topsLaunchHostFunc");
#ifdef RUNTIME_1_5_1_1016
  // 注册程序退出时的清理函数
  static bool cleanup_registered = false;
  if (!cleanup_registered) {
    std::atexit([]() { PythonCallbackWorker::destroyInstance(); });
    cleanup_registered = true;
  }
#endif
}

#ifdef RUNTIME_1_5_1_1016
// 静态成员定义
std::unique_ptr<PythonCallbackWorker> PythonCallbackWorker::instance = nullptr;
std::mutex PythonCallbackWorker::instance_mutex;
#endif
}  // namespace torch_gcu
