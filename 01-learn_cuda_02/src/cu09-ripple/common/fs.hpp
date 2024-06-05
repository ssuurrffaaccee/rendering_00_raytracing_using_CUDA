#ifndef FS_H
#define FS_H

#include <filesystem>
#include <string>
namespace fs {
inline bool isFile(const std::string& file_path) {
  return std::filesystem::is_regular_file(std::filesystem::path{file_path});
}
inline bool isDirectory(const std::string& file_path) {
  return std::filesystem::is_directory(std::filesystem::path{file_path});
}
inline bool exist(const std::string& file_path) {
  return std::filesystem::exists(std::filesystem::path{file_path});
}
inline bool checkExtension(const std::string& file_path, const std::string& ext) {
  return std::filesystem::path{file_path}.extension().string() == ext;
}
inline std::string absolute(const std::string& file_path) {
  return std::filesystem::absolute(std::filesystem::path{file_path}).string();
}
}  // namespace fs

#endif