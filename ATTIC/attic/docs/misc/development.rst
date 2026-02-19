Developing Pulserver
====================

Developement environment
------------------------

We recommend to use a virtual environement (e.g., Conda), and install pulserver and its dependencies in it.


Running tests
-------------

Test suite can be executed as:
    
.. code-block:: bash

    cd pulserver 
    pip install -e .[test]
    pytest .

Writing documentation
---------------------

Documentation is available online at https://github.com/INFN-MRI/pulserver

It can also be built locally as:

.. code-block:: bash

    cd pulserver
    pip install -e .[doc]
    python -m sphinx docs docs_build

To view the html doc locally you can use:

.. code-block:: bash

    python -m http.server --directory docs_build 8000

And visit `localhost:8000` on your web browser.
