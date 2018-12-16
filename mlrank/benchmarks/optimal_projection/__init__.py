from .optimal_projection_algorithm import (
    PCAWrap,
    TSNEWrap,
    ICAWrap,
    LLEWrap
)

from .optimal_projection_benchmark import (
    OptimalProjectionBenchmark
)

from .benchmarks import (
    lr_projection_benchmark,
    svc_projection_benchmark,
    rf_projection_benchmark
)

from .facade import OptimalProjectionBenchmarkFacade