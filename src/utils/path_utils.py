"""
Path resolution utilities for ClipperAI project.
Provides environment-friendly path resolution for cross-platform compatibility.
"""

import os
from pathlib import Path

def get_project_root():
    """
    Get the project root directory in a robust way.
    
    Returns:
        str: Path to the project root directory
    """
    # Try multiple methods to find project root
    candidates = []
    
    # Method 1: From current file location (if run from within project)
    try:
        # This file should be in src/utils, so go up 2 levels
        current_file = os.path.abspath(__file__)
        src_utils_dir = os.path.dirname(current_file)
        src_dir = os.path.dirname(src_utils_dir)
        project_root = os.path.dirname(src_dir)
        candidates.append(project_root)
    except:
        pass
    
    # Method 2: From current working directory
    try:
        cwd = os.getcwd()
        # Check if we're already in project root
        if os.path.exists(os.path.join(cwd, 'src')) and os.path.exists(os.path.join(cwd, 'bin')):
            candidates.append(cwd)
        else:
            # Check if we're in src directory
            if os.path.exists(os.path.join(cwd, '..', 'bin')):
                candidates.append(os.path.dirname(cwd))
    except:
        pass
    
    # Method 3: Look for common project markers
    try:
        cwd = os.getcwd()
        search_paths = [cwd] + [os.path.dirname(p) for p in [cwd] * 5]  # Go up max 5 levels
        
        for path in search_paths:
            if (os.path.exists(os.path.join(path, 'src')) and 
                os.path.exists(os.path.join(path, 'bin')) and
                (os.path.exists(os.path.join(path, 'templates')) or 
                 os.path.exists(os.path.join(path, 'README.md')))):
                candidates.append(path)
                break
    except:
        pass
    
    # Return the first valid candidate, or fallback to cwd
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    
    # Ultimate fallback
    return os.getcwd()

def get_bin_dir():
    """
    Get the bin directory path.
    
    Returns:
        str: Path to the bin directory
    """
    project_root = get_project_root()
    return os.path.join(project_root, 'bin')

def get_templates_dir():
    """
    Get the templates directory path.
    
    Returns:
        str: Path to the templates directory
    """
    project_root = get_project_root()
    return os.path.join(project_root, 'templates')

def get_output_dir():
    """
    Get the default output directory path.
    
    Returns:
        str: Path to the output directory
    """
    project_root = get_project_root()
    return os.path.join(project_root, 'output')

def setup_bin_path():
    """
    Add the bin directory to PATH if it exists.
    
    Returns:
        str: The bin directory path that was added to PATH (or None if not found)
    """
    bin_dir = get_bin_dir()
    
    if os.path.exists(bin_dir):
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path
        return bin_dir
    else:
        print(f"Warning: Bin directory not found at {bin_dir}")
        return None

def resolve_path(path_relative_to_project=None, absolute_path=None):
    """
    Resolve a path in an environment-friendly way.
    
    Args:
        path_relative_to_project (str): Path relative to project root
        absolute_path (str): Absolute path (takes precedence if provided)
    
    Returns:
        str: Resolved absolute path
    """
    if absolute_path:
        return os.path.abspath(absolute_path)
    
    if path_relative_to_project:
        project_root = get_project_root()
        return os.path.join(project_root, path_relative_to_project)
    
    # If neither provided, return project root
    return get_project_root()

# Convenience functions for common directories
def get_default_video_path():
    """Get default video path (podcast.mp4 in project root)."""
    return os.path.join(get_project_root(), "podcast.mp4")

def get_default_template_path(template_name="default"):
    """Get default template path."""
    return os.path.join(get_templates_dir(), template_name)
