# IR-General-library

IR-General-library is a header-only library providing general utilities for Information Retrieval (IR) tasks.

## Installation

This library is header-only. You can include it in your project by adding the `include` directory to your include paths.

### Using with CMake

To include this library in your CMake project, add the following lines to your `CMakeLists.txt`:

```cmake
add_subdirectory(IR-General-Library)
target_link_libraries(YourExecutable IRGeneralLibrary)
```

## Usage

Include the necessary headers in your source files:

```cpp
#include "IR-General-Library.hpp"
```
