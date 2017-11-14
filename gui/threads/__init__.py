"""This package contains the thread encapsulation of the :class:`~thermography.thermo_app.ThermoApp` class.
This encapsulation allows to work with :mod:`thermography` using a graphical interface."""

from .thermo_thread import ThermoGuiThread
from .thermo_thread_dataset_creation import ThermoDatasetCreationThread