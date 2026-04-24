//
// Created by xiang on 25-3-12.
//

#include "io/file_io.h"
#include <cstdlib>
#include <filesystem>

#include <glog/logging.h>

namespace lightning {
namespace {

namespace fs = std::filesystem;

fs::path NormalizePath(const fs::path& path) {
    std::error_code ec;
    auto normalized = fs::weakly_canonical(path, ec);
    if (!ec) {
        return normalized;
    }
    return path.lexically_normal();
}

bool IsColconWorkspaceRoot(const fs::path& path) {
    return fs::is_directory(path / "src") && (fs::is_directory(path / "install") || fs::is_directory(path / "build"));
}

fs::path ReferenceDirectory(const std::string& reference_path) {
    fs::path reference = reference_path.empty() ? fs::current_path() : fs::path(reference_path);
    if (reference.is_relative()) {
        reference = fs::absolute(reference);
    }

    std::error_code ec;
    if (!fs::is_directory(reference, ec)) {
        reference = reference.parent_path();
    }
    return NormalizePath(reference);
}

}  // namespace

bool PathExists(const std::string& file_path) {
    fs::path path(file_path);
    return fs::exists(path);
}

bool RemoveIfExist(const std::string& path) {
    if (PathExists(path)) {
        // LOG(INFO) << "remove " << path;
        int ret = std::system(("rm -f " + path).c_str());
        (void)ret;
        return true;
    }
    return false;
}

bool IsDirectory(const std::string& path) { return fs::is_directory(path); }

std::string FindWorkspaceRoot(const std::string& reference_path) {
    fs::path current = ReferenceDirectory(reference_path);
    while (!current.empty()) {
        if (IsColconWorkspaceRoot(current)) {
            return NormalizePath(current).string();
        }

        fs::path parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }
    return "";
}

std::string ResolveWorkspacePath(const std::string& path, const std::string& workspace_root) {
    if (path.empty()) {
        return path;
    }

    fs::path input(path);
    if (input.is_absolute()) {
        return NormalizePath(input).string();
    }

    fs::path base;
    if (workspace_root.empty()) {
        base = fs::current_path();
        LOG(WARNING) << "cannot find colcon workspace root; resolve relative path from current working directory: "
                     << path;
    } else {
        base = workspace_root;
    }

    return NormalizePath(base / input).string();
}

std::string ResolveWorkspacePathFrom(const std::string& path, const std::string& reference_path) {
    return ResolveWorkspacePath(path, FindWorkspaceRoot(reference_path));
}

}  // namespace lightning
