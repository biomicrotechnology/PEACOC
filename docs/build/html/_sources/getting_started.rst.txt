.. _getting_started:

***************
Getting started
***************

.. _download:

Download the toolbox
====================

.. _download_git:

Using git
----------------

Navigate to where you want to store the toolbox. For example::

    > cd PEACOC_tutorial/

Then clone the repository from git using the command-line::

    > git clone https://github.com/biomicrotechnology/PEACOC.git

The output on the command-line should look like this::

    Cloning into 'PEACOC'...
    remote: Enumerating objects: 75, done.
    remote: Counting objects: 100% (75/75), done.
    remote: Compressing objects: 100% (55/55), done.
    remote: Total 75 (delta 21), reused 72 (delta 18), pack-reused 0
    Receiving objects: 100% (75/75), 296.27 KiB | 2.77 MiB/s, done.
    Resolving deltas: 100% (21/21), done.


Now you have a sub-folder called *PEACOC* in the directory where you executed git clone.
*PEACOC* contains all the code.

.. _update_git:

Updating to a newer version
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can later update the toolbox to its newest version using git.
Navigate to the toolbox directory::

    > cd PEACOC

And then update to the newest version of the code::

    > git pull

But for now, let's stay one level above the folder *PEACOC*.

.. _download_zip:

Alternative: download archive
-----------------------------
If you do not want to use git, you can also download the `zip archive <https://github.com/biomicrotechnology/PEACOC/archive/master.zip>`_ from github.
Extract this to a place where you want to store the toolbox.


.. _virtual_env:

Set up a virtual environment (recommended)
==========================================

I strongly recommened installing all requirements for *PEACOC* in a new virtual environment and always run the program from there.
This keeps all dependencies nice and tidy and prevents that stuff does not work anymore due to an annoying update.
Here is how to set up a virtual environment.
To use virtual environments, use this command on Linux::

    > pip install virtualenvwrapper

Or this on for a Windows machine::

    > pip install virtualenvwrapper-win


Then create a virtual environment and let's name it *peacoc*::

    > mkvirtualenv --python=python3 peacoc

To enter this environment, type::

    > workon peacoc


You can leave this environment by typing::

    > deactivate

But for now, let's stay in *peacoc*.


.. _requirements:

Install requirements
====================

If you do not have python 3 (code also works with python version >=2.5) and pip 3 on your computer.
You can find instructions about how to install these two at `python <https://realpython.com/installing-python/>`_ and `pip <https://pip.pypa.io/en/stable/>`_.

If you decided to use a virtual environment for EA detection, make sure you are in it::

    > workon peacoc

Install the requirements::

    > pip3 install -r PEACOC/requirements.txt

