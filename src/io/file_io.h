//
// Created by xiang on 25-3-12.
//

#ifndef LIGHTNING_FILE_IO_H
#define LIGHTNING_FILE_IO_H

#include <string>

#include "common/eigen_types.h"
#include "common/std_types.h"

namespace lightning {

/**
 * 检查某个路径是否存在
 * @param file_path 路径名
 * @return true if exist
 */
bool PathExists(const std::string& file_path);

/**
 * 若文件存在，则删除之
 * @param path
 * @return
 */
bool RemoveIfExist(const std::string& path);

/**
 * 判断某路径是否为目录
 * @param path
 * @return
 */
bool IsDirectory(const std::string& path);

/**
 * 从参考路径向上查找colcon workspace根目录
 * @param reference_path 通常为配置文件路径
 * @return workspace根目录，找不到时返回空字符串
 */
std::string FindWorkspaceRoot(const std::string& reference_path);

/**
 * 将相对路径解析到workspace根目录下，绝对路径保持绝对
 * @param path 待解析路径
 * @param workspace_root workspace根目录，空时回退到当前工作目录
 * @return 规范化后的绝对路径
 */
std::string ResolveWorkspacePath(const std::string& path, const std::string& workspace_root);

/**
 * 从参考路径查找workspace根目录后解析路径
 * @param path 待解析路径
 * @param reference_path 通常为配置文件路径
 * @return 规范化后的绝对路径
 */
std::string ResolveWorkspacePathFrom(const std::string& path, const std::string& reference_path);

}

#endif  // LIGHTNING_FILE_IO_H
