.. PEACOC documentation master file, created by
   sphinx-quickstart on Sun Jan 17 10:46:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Detecting epileptiform activity with PEACOC
===========================================

PEACOC stands for **P**\atterns of **E**\pileptiform **A**\ctivity: a **CO**\ntinuum-based **C**\lassification.
It is s a tool that allows you to automatically **detected and classify epilepitiform activity (EA) patterns**.
It transforms a local field potential (LFP) recording into a sequence of classified spike bursts and solitary spikes.
The tool is python-based and you can execute it via the command-line.

The EA patterns PEACOC identifies are:

- *high-load bursts*, these comprise large behavioral seizures and paroxysmal and seizure-like activity (e.g. high-voltage
sharp waves and hyperparoxysmal discharges)

- *medium-load bursts*, a category bridging the continuum between high-load and low-load bursts

- *low-load bursts*, which are looser trains of epileptiform spikes

- *solitary spikes*, i.e. epileptiform spikes that are not part of a burst

Three **major analysis steps** are involved: I) preprocessing (resampling, assigning the polarity,
semiautomatic artifact detection), II) transforming the LFP into a train of epileptiform spikes, and III)
transforming the spike train into a sequence of classified spike bursts.

.. image:: _static/screenshots/fig6_codeflow.png
    :width: 700
    :align: center

The tool was originally developed for the **intrahippocampal kainate mouse model**, but we have seen promising results for
other animal models of epilepsy as well. It has been used in several publications, but you will find the method best described
in *Heining et al. (2021)_*.


In this documentation you will learn how to :ref:`set everything up <getting_started>`, :ref:`set the parameters <setting_parameters>`,
run the :ref:`preprocessing <preprocessing>` and the :ref:`actual analyses<LFP_to_bursts>`, :ref:`harvest the results <access_results>`,
and :ref:`retrieve metadata and diagnostic metrics <access_metadata>`.
You can download the example data and the files successively generated in this tutorial from *here_*.
If can also download a *pdf-version_* of this documentation.


.. todo::

    1) Make a link to Heining et al. 2021 once the paper is out.
    2) Make a link to the tutorial data file.
    3) Make a link to the pdf.



.. toctree::
   :maxdepth: 2

   getting_started.rst

   setting_parameters.rst

   preprocessing.rst

   LFP_to_bursts.rst

   output.rst

   access_results.rst

   access_metadata.rst


Todolist
========

.. todolist::



