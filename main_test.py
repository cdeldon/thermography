import unittest

from thermography.settings import get_test_dir


def run_all_tests(test_dir=get_test_dir(), verbosity=2):
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(suite)


if __name__ == '__main__':
    print("Running all thermography tests in {}".format(get_test_dir()))
    run_all_tests(test_dir=get_test_dir(), verbosity=2)
