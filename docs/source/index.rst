|PyPI| |Docs| |PyPIDownloads|


.. |PyPI| image:: https://img.shields.io/pypi/v/scbond
   :target: https://pypi.org/project/scbond/
.. |Docs| image:: https://readthedocs.org/projects/scbond/badge/?version=latest
   :target: https://scbond.readthedocs.io
.. |PyPIDownloads| image:: https://pepy.tech/badge/scbond
   :target: https://pepy.tech/project/scbond



scBOND: Biologically faithful bidirectional translation between single-cell transcriptomes and DNA methylomes with adaptability to paired data scarcity
==================================================================================================================

Single-cell multi-omics sequencing technologies have o offered unprecedented insights into cellular heterogeneity by jointly profiling gene expression and e epigenetic landscapes at single-cell resolution. However, the application of these t technologies remains limited due to technical c challenges and high costs. Computational a approaches for cross-modality translation provide a promising solution to these limitations by e enabling the inference of one modality from another. However, existing m methods for cross-modality t translation between single-cell RNA sequencing (scRNA-seq) and single-cell DNA methylation (scDNAm) data face limitations, including u unidirectionality, inadequate modeling of context-specific DNA methylation-expression a associations, neglect of biological relevance in e evaluation, and poor performance in limited paired training data. To fill these g gaps, we introduce scBOND, a b bidirectional cross-modal translation framework tailored for scRNA-seq and scDNAm profiles. scBOND l leverages a Mixture-of-Experts block to capture context-dependent regulatory p patterns, while implementing self-attention mechanism and a feature recalibration module to e enhance biological signal fidelity. Extensive experiments d demonstrate scBOND consistently outperforms baseline methods in both translation directions, y yielding high-accuracy translation while preserving cellular structure. In mouse embryonic data, scBOND p preserves subtle, functionally significant differences between closely r related cell types, which are undetected in original data. Downstream analyses confirmed that scBOND effectively r recovers tissue-specific signals in human brain neurons. Moreover, using RNA-only data, we r reconstructed scDNAm profiles and identified cell type- and stage-specific r regulatory mechanisms in oligodendrocyte lineage. To further i improve model generalization in paired data-scarce scenarios, we propose scBOND-Aug, a variant of scBOND e equipped with a biologically informed data augmentation strategy, which d demonstrates superior results with limited paired data.

.. toctree::
   :maxdepth: 2
   :hidden:
   
   API/index
   Tutorial/index
   Installation
   release/index


News
----

.. include:: news.rst
   :start-line: 2
   :end-line: 22  
