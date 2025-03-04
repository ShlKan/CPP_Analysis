
import lit.formats

config.name = "IRAnalysis"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root, 'test')

config.substitutions.append(('%cpp_analysis',
    os.path.join(config.my_obj_root, 'IRAnalysis/Driver/cpp_analysis')))

cpp_analysis_args = [
    "-internal-isystem",
    "/Users/shuanglong.kan/MyProjects/install-llvm/bin/../include/c++/v1",
    "-internal-isystem",
    "/Users/shuanglong.kan/MyProjects/install-llvm/lib/clang/20/include",
    "-internal-externc-isystem",
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/local/include",
    "-internal-externc-isystem",
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include",
    "-fgnuc-version=4.2.1",
    "-x",
    "c++",
    "-fcxx-exceptions",
    "-I/Users/shuanglong.kan/MyProjects/systemc/include",
    "-sysir"
]

cpp_analysis_args_str = ' '.join(cpp_analysis_args)

config.substitutions.append(('%args', cpp_analysis_args_str))