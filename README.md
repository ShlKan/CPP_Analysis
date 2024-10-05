
# CPP_Analysis: A Analyzer of c or c++

The tool is built on [ClangIR](https://llvm.github.io/clangir//).

## 1.  Build
CPP_Analysis depends on LLVM. So before building CPP_Analysis, LLVM should be build and add the find_package CMAKE_PREFIX_PATH.

### 1.1 Building LLVM
The important thing the following parameter: **Add LLVM_ENABLE_PROJECTS="clang;mlir"**. Because we need to use clang frontend and mlir.

For instance, an command to build llvm could be:
```
cmake -DCMAKE_BUILD_TYPE=Debug
      -DLLVM_ENABLE_PROJECTS=clang;mlir
      -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE 
      -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang
      -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++
      --no-warn-unused-cli 
      -S/Users/shuanglong/projects/llvm-project/llvm 
      -B/Users/shuanglong/projects/llvm-project/build 
      -G Ninja 
      path/to/cmakefile
```

**Attention:** The llvm now needs to change a little bit see [Fix issues of ClangIR](https://github.com/ShlKan/CPP_Analysis/issues/1)

### 1.2 Building CPP_Analysis
You can use the following command to compile CPP_Analysis project.
```
cmake -DCMAKE_BUILD_TYPE=Debug 
	  -DCMAKE_PREFIX_PATH=/path/to/llvm-project/build 
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE 
	  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang 
	  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ 
	  --no-warn-unused-cli 
	  -S/Users/shuanglong/projects/CPP_ANALYSIS 
	  -B/Users/shuanglong/projects/CPP_Analysis/build 
	  -G Ninja
```

`CMAKE_PREFIX_PATH` must be the path to the llvm build directory you just compiled.




