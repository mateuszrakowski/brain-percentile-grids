import logging
import warnings

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

logger = logging.getLogger(__name__)


class REnvironmentError(Exception):
    """
    Exception raised when R environment cannot be initialized.

    This exception is raised when there are issues with R installation,
    missing R packages, or other R environment configuration problems.
    """

    pass


class REnvironment:
    """
    Singleton class to manage the R environment connection.

    Ensures that R packages are imported and the environment is
    initialized only once during the application's lifecycle.

    Attributes
    ----------
    base : rpackages.Package
        R base package.
    stats : rpackages.Package
        R stats package.
    grDevices : rpackages.Package
        R graphics devices package.
    gamlss_r : rpackages.Package
        GAMLSS R package.
    gamlss_dist : rpackages.Package
        GAMLSS distributions package.
    pandas2ri : module
        rpy2 pandas to R conversion module.
    localconverter : function
        Local context converter for rpy2.
    robjects : module
        rpy2 robjects module.

    Raises
    ------
    REnvironmentError
        If R or required packages are not available.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(REnvironment, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._init_error: Exception | None = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.base = rpackages.importr("base")
                self.stats = rpackages.importr("stats")
                self.grDevices = rpackages.importr("grDevices")
                self.gamlss_r = rpackages.importr("gamlss")
                self.gamlss_dist = rpackages.importr("gamlss.dist")

                self.pandas2ri = pandas2ri
                self.localconverter = localconverter
                self.robjects = robjects

                self.pandas2ri.activate()

                logger.info("Successfully initialized R environment.")
                self._initialized = True
            except Exception as e:
                self._init_error = e
                self._initialized = True  # Prevent retry loops
                logger.error(f"Failed to initialize R environment: {e}")
                raise REnvironmentError(
                    f"R environment initialization failed: {e}"
                ) from e

    @property
    def is_available(self) -> bool:
        """
        Check if R environment was successfully initialized.

        Returns
        -------
        bool
            True if R environment is available, False otherwise.
        """
        return self._initialized and self._init_error is None


def get_r_environment() -> REnvironment:
    """
    Get the R environment singleton.

    Returns
    -------
    REnvironment
        The initialized R environment.

    Raises
    ------
    REnvironmentError
        If R environment is not available.
    """
    env = REnvironment()
    if not env.is_available:
        raise REnvironmentError("R environment is not available")
    return env


def check_r_environment() -> bool:
    """
    Check if R environment is properly available.

    Returns
    -------
    bool
        True if R environment is available, False otherwise.
    """
    try:
        return REnvironment().is_available
    except REnvironmentError:
        return False
