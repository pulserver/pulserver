Pulserver
=========
|Coverage| |CI/CD| |License| |Codefactor| |Sphinx| |PyPi| |Black| |PythonVersion|

.. |Coverage| image:: https://codecov.io/gh/INFN-MRI/pulserver/graph/badge.svg?token=qtB53xANwI 
 :target: https://codecov.io/gh/INFN-MRI/pulserver

.. |CI/CD| image:: https://github.com/INFN-MRI/pulserver/workflows/CI-CD/badge.svg
   :target: https://github.com/INFN-MRI/pulserver

.. |License| image:: https://img.shields.io/github/license/INFN-MRI/pulserver
   :target: https://github.com/INFN-MRI/pulserver/blob/main/LICENSE.txt

.. |Codefactor| image:: https://www.codefactor.io/repository/github/INFN-MRI/pulserver/badge
   :target: https://www.codefactor.io/repository/github/INFN-MRI/pulserver

.. |Sphinx| image:: https://img.shields.io/badge/docs-Sphinx-blue
   :target: https://infn-mri.github.io/pulserver

.. |PyPi| image:: https://img.shields.io/pypi/v/pulserver
   :target: https://pypi.org/project/pulserver

.. |Black| image:: https://img.shields.io/badge/style-black-black

.. |PythonVersion| image:: https://img.shields.io/badge/Python-%3E=3.10-blue?logo=python&logoColor=white
   :target: https://python.org

Pulserver is a high-level wrapper over PyPulseq. It is designed to 
improve MR sequence code organization and provide a bridge
between different Pulseq interpreter representations while providing
a familiar interface to PyPulseq user.

It also provide a server-based interface to enable online sequence design
routines, e.g., using the MR scanner as a client.

Features
--------



Installation
------------
Pulserver can be installed via pip:

.. code-block:: bash

    pip install pulserver

Development
-----------
If you want to modifiy the Pulserver code base:

.. code-block:: bash

    git clone https://github.com/INFN-MRI/pulservert.git
    pip install -e ./pulserver

Usage
-----

TODO: add quick example

Testing
-------
To run the tests, execute the following command in the terminal:

.. code-block:: bash

     pytest .

License
-------
This project is licensed under the MIT License.

Contributing
------------
Contributions are welcome! Please fork the repository and submit a pull request.
