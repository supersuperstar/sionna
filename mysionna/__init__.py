#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""This is the Sionna library.
"""

__version__ = '0.16.1'

from .constants import *
from . import rt
from .config import *

# Instantiate global configuration object
config = Config()
