.. module:: scBond
.. automodule:: scBond
   :noindex:

API
====


Import scBond::

   import scBond

bond
-------------------------
.. module::scBond.bond
.. currentmodule::scBond
.. autosummary::
    :toctree: .

    bond.Bond
    bond.Bond.load_data
    bond.Bond.data_preprocessing
    bond.Bond.augmentation
    bond.Bond.construct_model
    bond.Bond.train_model
    bond.Bond.test_model
    bond.Bond.predict_single_rna
    bond.Bond.predict_single_methylation
    bond.Bond.predict_single_modal


calculate_cluster
-------------------------
.. module::scBond.calculate_cluster
.. currentmodule::scBond
.. autosummary::
    :toctree: .

    calculate_cluster.calculate_cluster_index


data_processing
-------------------------
.. module::scBond.data_processing
.. currentmodule::scBond
.. autosummary::
    :toctree: .

    data_processing.add_methylation_noise
    data_processing.imputation_met
    data_processing.MET_data_preprocessing
    data_processing.RNA_data_preprocessing


draw_cluster
-------------------------
.. module::scBond.draw_cluster
.. currentmodule::scBond
.. autosummary::
    :toctree: .

    draw_cluster.draw_tsne
    draw_cluster.draw_umap


train_model
-------------------------
.. module::scBond.train_model
.. currentmodule::scBond
.. autosummary::
    :toctree: .

    train_model.Model
    train_model.Model.predict_single_methylation
    train_model.Model.predict_single_rna
    train_model.Model.test
    train_model.Model.train
