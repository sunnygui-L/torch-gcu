#include <c10/util/Logging.h>

#include <cstdlib>
#include <iostream>
#include <string>

/*
LogBlock is used to temply set log level in a code block.
Valid values for block_log_level are `INFO`, `WARNING`, `ERROR`,
and `FATAL` or their numerical equivalents `0`, `1`, `2`, and `3`.
example
{
    LogBlock error_block("ERROR");
    ...
}
*/
class LogBlock {
 public:
  explicit LogBlock(const char* block_log_level) {
    const char* prev_log_level = std::getenv("TORCH_CPP_LOG_LEVEL");
    if (prev_log_level != nullptr) {
      _prev_log_level = std::string(prev_log_level);
    } else {
      _prev_log_level = "";
    }
    setenv("TORCH_CPP_LOG_LEVEL", block_log_level, 1);
    c10::initLogging();
  }

  LogBlock(const LogBlock&) = delete;
  LogBlock(LogBlock&&) noexcept = delete;
  LogBlock& operator=(const LogBlock&) = delete;
  LogBlock& operator=(LogBlock&&) noexcept = delete;

  ~LogBlock() {
    setenv("TORCH_CPP_LOG_LEVEL", _prev_log_level.c_str(), 1);
    c10::initLogging();
  }

 private:
  std::string _prev_log_level;
};

class SuppressTorchWarn : public LogBlock {
 public:
  explicit SuppressTorchWarn() : LogBlock("ERROR") {}
};