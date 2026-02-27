from .case import (
	addModuleCleanup as addModuleCleanup, expectedFailure as expectedFailure, FunctionTestCase as FunctionTestCase,
	skip as skip, skipIf as skipIf, SkipTest as SkipTest, skipUnless as skipUnless, TestCase as TestCase)
from .loader import defaultTestLoader as defaultTestLoader, TestLoader as TestLoader
from .main import main as main, TestProgram as TestProgram
from .result import TestResult as TestResult
from .runner import TextTestResult as TextTestResult, TextTestRunner as TextTestRunner
from .signals import (
	installHandler as installHandler, registerResult as registerResult, removeHandler as removeHandler,
	removeResult as removeResult)
from .suite import BaseTestSuite as BaseTestSuite, TestSuite as TestSuite
from unittest.async_case import *
import sys

if sys.version_info >= (3, 11):
    from .case import doModuleCleanups as doModuleCleanups, enterModuleContext as enterModuleContext

__all__ = [
    "FunctionTestCase",
    "IsolatedAsyncioTestCase",
    "SkipTest",
    "TestCase",
    "TestLoader",
    "TestResult",
    "TestSuite",
    "TextTestResult",
    "TextTestRunner",
    "addModuleCleanup",
    "defaultTestLoader",
    "expectedFailure",
    "installHandler",
    "main",
    "registerResult",
    "removeHandler",
    "removeResult",
    "skip",
    "skipIf",
    "skipUnless",
]

if sys.version_info < (3, 13):
    from .loader import findTestCases as findTestCases, getTestCaseNames as getTestCaseNames, makeSuite as makeSuite

    __all__ += ["findTestCases", "getTestCaseNames", "makeSuite"]

if sys.version_info >= (3, 11):
    __all__ += ["doModuleCleanups", "enterModuleContext"]

if sys.version_info < (3, 12):
    def load_tests(loader: TestLoader, tests: TestSuite, pattern: str | None) -> TestSuite: ...

def __dir__() -> set[str]: ...
