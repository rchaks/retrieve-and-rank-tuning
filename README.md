# rnr-debugging-scripts

This repository contains helper scripts and examples that demonstrate
* how to use the `Retrieve And Rank` and `Discovery` watson services
* how to evaluate the quality of retrieval performance with your own
  data

This is meant for users who might want to use the APIs to manage 
 their services (rather than the [GUI tools](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/ranker_tooling.html))
  for more control or debugging.  All interaction between my python
  code and the services take place using the [python client libraries](https://github.com/watson-developer-cloud/python-sdk)
  provided by IBM.

For more information about `Retrieve And Rank`, see the [detailed documentation](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/index.html).

For more information about `Discovery`, see the [detailed documentation](https://www.ibm.com/watson/developercloud/doc/discovery/index.html).

# Getting Started
## 0. Clone the repository
`git clone...`

## 1. Setup Python Environment
1. Ensure installation of [Python3.5](https://www.python.org/downloads/release/python-353/)
(presumably any Python 3.x version should work, but has only been tested with 3.5) 

2. Install [pip](https://pip.pypa.io/en/stable/installing/)
    - If you already have it installed, you might still
  want to check if you've updated to the latest installation `pip install --upgrade pip`

3. Install [virtualenv](https://virtualenv.pypa.io/en/stable/installation/)

4. (Optional) Initialize a clean virtualenv:
    - `virtualenv {location of virtualenv}`
    - `source {location of virtualenv}/bin/activate`

## 2. Get Watson Credentials from Bluemix
1. Ensure that you have a [Bluemix account](https://console.ng.bluemix.net/registration/): 
 You will use Bluemix to provision Watson services (even the free-tier 
 user account priviledges should be enough for to try out these 
 scripts)

2. Depending on which service(s) you wish to try out, provision a
 [Retrieve And Rank](https://console.bluemix.net/catalog/services/retrieve-and-rank?env_id=ibm:yp:eu-gb)
  and/or [Discovery](https://console.bluemix.net/catalog/services/discovery?env_id=ibm:yp:eu-gb)
   service from bluemix. Alternatively, you can use the [cloud foundary 
   tools](https://github.com/cloudfoundry/cli#downloads) 
   to provision services using command line utilities.

3. Edit the `config/config.ini` file with your credentials for
  one or more services. Should look something like this:
  ```
[RetrieveAndRank]
# Credentials from Bluemix
user=7eeeee-30000-4eee-beee-92313faea
password=QQQQQQQQQy

[Discovery]
user=7eeeee-3012310-4eee-beee-92313faea
password=QQQQQQQ123
  ```
 
## Install the library

1. Install the helper scripts as a python library 
(along necessary dependencies): `pip install -e .` 
from inside the main directory.

2. If you want to test that the scripts are working as expected, you
can also kick off the unit tests by running: `py.test tests`. 
> **Warning:** These unit tests will use your credentials to actually
call bluemix.  If unit tests are interrupted before completion, 
they may leave data on 'UnitTest' clusters/collections/rankers around.
 And the tests take a loooong time to execute on account of 
 having to reach out to the watson services.
 
# Examples

Sample usage of the scripts to do a number of tasks are available
in the [examples directory as Jupyter notebooks](examples).

You can just view these in github (it has
 sample output), but if you wish to run them locally, you will need 
 to install additional dependencies: `pip install -r requirements-examples.txt`

Though the Jupyter notebooks show usage from inside Python, the scripts
inside [rnr_debug_helpers](rnr_debug_helpers) are also command line 
 friendly.  For usage info run:
  `python rnr_debug_helpers <script_name.py> --help`

# License

  This sample code is licensed under Apache 2.0.
  Full license text is available in [LICENSE](LICENSE).
