#!/usr/bin/env bash

# Copyright (c) 2020  Thiago Lopes da Silva

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# stop current script if any error happens
set -e

# get parent folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
PARENT_DIRECTORY="${DIR%/*}"

# go to parent folder
cd ${PARENT_DIRECTORY}

# run all unit tests
#python -m pytest --import-mode=append tests/
python -m unittest app.py
# python -m unittest discover
