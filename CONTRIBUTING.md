## Introduction

Thanks for considering contributing!  Hopefully this repo will continue to provide a set of helper scripts for common
bits of analysis and setup for question answering (or information retrieval) tasks using the Watson APIs.

Additional example/samples, scripts as well as documentation are all welcome.  I do hope to stick to the Long Tail 
Question Answering task - so only evaluation and debugging of Retrieve-and-Rank or Discovery APIs.

## Questions/Issues

If you are having difficulties with the underlying the IBM Watson Services or python-sdks that I'm making a use of, then consider asking your questions on their [dW Answers][dw] or [Stack Overflow][stackoverflow] forums.

If the questions/issues are specific to the scripts/examples in this repository, feel free to submit them as a [bug report or issue](https://github.com/rchaks/retrieve-and-rank-tuning/issues).

## Pull Requests

If you want to contribute to the repository, here's a quick guide:
  1. Fork the repository
  2. develop and test your code changes with [pytest].
    * Respect the usual [PEP8 standards](https://www.python.org/dev/peps/pep-0008/).
    * Only use spaces for indentation.
    * Create minimal diffs - disable on save actions like reformat source code or organize imports. If you feel the source code should be reformatted create a separate PR for this change.
    * Check for unnecessary whitespace with git diff --check before committing.
  3. Make the test pass
  > Unfortunately the tests require bluemix credentials currently (haven't spent the time to build proper mocks).  So you'll have to update the config with valid credentials.  The free versions of the services will suffice for the tests.
  4. Commit your changes
  5. Push to your fork and submit a pull request to the `master` branch

## Setup & Running the tests

Refer to the main [README.md](README.md) for details.

## Additional Resources
+ [General GitHub documentation](https://help.github.com/)
+ [GitHub pull request documentation](https://help.github.com/send-pull-requests/)

[dw]: https://developer.ibm.com/answers/questions/ask/?topics=watson
[stackoverflow]: http://stackoverflow.com/questions/ask?tags=ibm-watson
[pytest]: http://pytest.org/latest/
[virtualenv]: http://virtualenv.readthedocs.org/en/latest/index.html
