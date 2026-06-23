#include <gtest/gtest.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/torch.h>
#include <torch_gcu.h>

#include <thread>
#include <vector>

const torch::DeviceType GCU_DEV = torch::kPrivateUse1;

class Resnet50Test : public testing::Test {
 protected:
  Resnet50Test() {}
  void SetUp() override { torch::manual_seed(1023); }
};

TEST_F(Resnet50Test, SingleThreadSingleStream) {
  torch::NoGradGuard no_grad;
  auto module = torch::jit::load("resnet50_traced.pt");
  module.eval();
  auto cpu_inputs = torch::randn({10, 3, 224, 224});
  auto cpu_outputs = module.forward({cpu_inputs}).toTensor();

  module.to(GCU_DEV);
  module.eval();
  auto dev_inputs = cpu_inputs.to(GCU_DEV);
  EXPECT_TRUE(torch::allclose(dev_inputs.cpu(), cpu_inputs));

  auto dev_outputs = module.forward({dev_inputs}).toTensor();
  EXPECT_TRUE(torch::allclose(dev_outputs.cpu(), cpu_outputs, 1e-1, 1e-1));
}

TEST_F(Resnet50Test, MultiThreadSingleStream) {
  torch::NoGradGuard no_grad;
  const int worker_num = 10;
  auto module = torch::jit::load("resnet50_traced.pt");
  module.eval();

  // cpu module run
  std::vector<torch::Tensor> cpu_inputs;
  std::vector<torch::Tensor> cpu_outputs;
  for (int i = 0; i < worker_num; ++i) {
    auto cpu_in_data = torch::randn({1, 3, 224, 224});
    cpu_inputs.push_back(cpu_in_data);

    auto output = module.forward({cpu_in_data}).toTensor();
    cpu_outputs.push_back(std::move(output));
  }

  // gcu module run
  module.to(GCU_DEV);
  std::vector<torch::Tensor> dev_outputs(
      worker_num, torch::empty_like(cpu_outputs[0],
                                    torch::TensorOptions().device(GCU_DEV)));
  std::vector<std::thread> workers;
  for (int i = 0; i < worker_num; ++i) {
    auto worker = std::thread([&module, &cpu_inputs, i, &dev_outputs]() {
      auto output = module.forward({cpu_inputs[i].to(GCU_DEV)}).toTensor();
      dev_outputs[i] = std::move(output);
    });
    workers.push_back(std::move(worker));
  }

  for (int i = 0; i < worker_num; ++i) {
    workers[i].join();
  }

  // check
  for (int i = 0; i < worker_num; ++i) {
    EXPECT_TRUE(
        torch::allclose(dev_outputs[i].cpu(), cpu_outputs[i], 1e-1, 1e-1));
  }
}

TEST_F(Resnet50Test, MultiThreadMultiStream) {
  torch::NoGradGuard no_grad;
  const int worker_num = 10;
  auto module = torch::jit::load("resnet50_traced.pt");
  module.eval();

  // cpu module run
  std::vector<torch::Tensor> cpu_inputs;
  std::vector<torch::Tensor> cpu_outputs;
  for (int i = 0; i < worker_num; ++i) {
    auto cpu_in_data = torch::randn({1, 3, 224, 224});
    cpu_inputs.push_back(cpu_in_data);

    auto output = module.forward({cpu_in_data}).toTensor();
    cpu_outputs.push_back(std::move(output));
  }

  // gcu module run
  module.to(GCU_DEV);
  std::vector<torch::Tensor> dev_outputs(
      worker_num, torch::empty_like(cpu_outputs[0],
                                    torch::TensorOptions().device(GCU_DEV)));
  std::vector<std::thread> workers;
  for (int i = 0; i < worker_num; ++i) {
    auto worker = std::thread([&module, &cpu_inputs, i, &dev_outputs]() {
      auto stream = torch_gcu::getStreamFromPool();
      torch_gcu::GCUStreamGuard guard(stream);
      auto output = module.forward({cpu_inputs[i].to(GCU_DEV)}).toTensor();
      dev_outputs[i] = std::move(output);
      stream.synchronize();
    });
    workers.push_back(std::move(worker));
  }

  for (int i = 0; i < worker_num; ++i) {
    workers[i].join();
  }

  // check
  for (int i = 0; i < worker_num; ++i) {
    EXPECT_TRUE(
        torch::allclose(dev_outputs[i].cpu(), cpu_outputs[i], 1e-1, 1e-1));
  }
}