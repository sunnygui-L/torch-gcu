/*
 * Copyright 2021-2023 Enflame. All Rights Reserved.
 */
#include "gcu/sys_util.h"

#include <c10/util/Exception.h>
#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>

#include "gcu/gcu_macros.h"
#include "gcu/logging.h"

GCU_TORCH_GCU_UTIL_NS_BEGIN

std::string GetEnvString(const char* name, const std::string& defval) {
  const char* env = std::getenv(name);
  auto val = env != nullptr ? env : defval;
  return val;
}

int64_t GetEnvInt(const char* name, int64_t defval) {
  const char* env = std::getenv(name);
  auto val = env != nullptr ? std::atol(env) : defval;
  return val;
}

double GetEnvDouble(const char* name, double defval) {
  const char* env = std::getenv(name);
  auto val = env != nullptr ? std::atof(env) : defval;
  return val;
}

bool GetEnvBoolNoPrint(const char* name, bool defval) {
  const char* env = std::getenv(name);
  bool val = false;
  if (env == nullptr) {
    val = defval;
  } else if (std::strcmp(env, "true") == 0) {
    val = true;
  } else if (std::strcmp(env, "false") == 0) {
    val = false;
  } else {
    val = std::atoi(env) != 0;
  }
  return val;
}

TORCH_GCU_API bool GetEnvBool(const char* name, bool defval) {
  bool val = GetEnvBoolNoPrint(name, defval);
  return val;
}

int64_t NowNs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

std::string GetTimeStamp() {
  std::time_t timestamp = time(0);

  const char* fmt = "%Y%m%d_%H%M%S";

  // Get the current system timezone information
  struct tm t;
  if (localtime_r(&timestamp, &t)) {
    // Adjust the time to be in Shanghai's time zone
    t.tm_hour += 8;

    // Format the adjusted time into a string using the specified format
    char buffer[1024];
    size_t len = strftime(buffer, sizeof(buffer), fmt, &t);
    return std::string(buffer, len);
  } else {
    return "Failed to convert timestamp.";
  }
}

std::string GetTimeStampMillis() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&t), "%Y%m%d%H%M%S") << '.'
     << std::setfill('0') << std::setw(3) << (millis % 1000);
  return ss.str();
}

void CreatDirIfNotExist(const std::string& path) {
  size_t pos = 0;
  std::string dir = path;
  if (dir[dir.size() - 1] != '/') {
    dir += '/';
  }
  while ((pos = dir.find_first_of('/', pos + 1)) != std::string::npos) {
    struct stat st_sub;
    memset(&st_sub, 0, sizeof(st_sub));
    if (stat(dir.substr(0, pos).c_str(), &st_sub) == -1) {
      if (mkdir(dir.substr(0, pos).c_str(),
                S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        TORCH_WARN("WARRING: mkdir ", dir.substr(0, pos), " fail! ",
                   strerror(errno));
      }
    }
  }
  struct stat st;
  memset(&st, 0, sizeof(st));
  if (stat(path.c_str(), &st) == -1) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
      TORCH_WARN("WARRING: mkdir ", path, " fail! ", strerror(errno));
    }
  }
}

void SaveJsonToFile(const std::string& file, Json::Value value) {
  std::ofstream fout(file, std::ios::out);
  if (fout.is_open()) {
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "   ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(value, &fout);
    fout.close();
    PTDLOG(TORCH_GCU) << "Success creat json file: " << file;
  } else {
    PTDLOG(TORCH_GCU) << "Cannot open json file: " << file;
  }
}

GCU_TORCH_GCU_UTIL_NS_END
