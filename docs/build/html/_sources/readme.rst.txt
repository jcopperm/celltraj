========
celltraj
========


.. image:: https://img.shields.io/pypi/v/celltraj.svg
        :target: https://pypi.python.org/pypi/celltraj

--------------------------------------------------------------------------------------
A toolset for the modeling and analysis of single-cell trajectories.
--------------------------------------------------------------------------------------

Key Features
------------
- **Single-Cell Trajectory Analysis**: Leverages time-lapse imaging data to construct detailed trajectories of single-cell behavior, capturing changes in morphology and motility.
- **Morphodynamical State Decomposition**: Utilizes data-driven methods to define and analyze cell states based on dynamic cellular features, providing insights into cell state transitions.
- **Dynamical Modeling**: Implements MSMs and Koopman operator-based approaches to kinetically characterize cell state transitions and generate embeddings for visualizing cell dynamics.
- **Integration with Molecular Data**: Maps live-cell imaging data to gene expression profiles, enabling predictions of RNA transcript levels based on cell state dynamics.
- **Tutorials**: Includes jupyter-notebooks with links to Zenodo repositories with downloadable data to guide users through the process of trajectory embedding and MMIST (Molecular and Morphodynamics-Integrated Single-cell Trajectories).

References
----------
- Copperman, Jeremy, Sean M. Gross, Young Hwan Chang, Laura M. Heiser, and Daniel M. Zuckerman. “Morphodynamical cell state description via live-cell imaging trajectory embedding.” Communications Biology 6, no. 1 (2023): 484.
- Copperman, Jeremy, Ian C. Mclean, Sean M. Gross, Young Hwan Chang, Daniel M. Zuckerman, and Laura M. Heiser. “Single-cell morphodynamical trajectories enable prediction of gene expression accompanying cell state change.” bioRxiv (2024): 2024-01.

License
-------
Free software: MIT license

Documentation
-------------
- `celltraj documentation <https://jcopperm.github.io/celltraj>`_

Tutorials
---------
- `Morphodynamical Trajectory Embedding Tutorial <https://github.com/jcopperm/celltraj/blob/main/tutorials/trajectory_embedding.ipynb>`_
- `MMIST: Molecular and Morphodynamics-Integrated Single-cell Trajectories Tutorial <https://github.com/jcopperm/celltraj/blob/main/tutorials/mmist.ipynb>`_

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
