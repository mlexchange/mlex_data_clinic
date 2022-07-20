# mlex_data_clinic

## Description
This app provides a training/testing platform for latent space exploration with
unsupervised deep-learning approaches.

## Running as a standalone application
First, let's install docker:

* https://docs.docker.com/engine/install/

Next, let's start its dependencies (in the order):
* [mlex_computing_api](https://github.com/mlexchange/mlex_computing_api)
* [mlex_content_registry](https://github.com/mlexchange/mlex_content_registry)

Before moving to the next step, please make sure that the computing API and the content 
registry are up and running. For more information, please refer to their respective 
README files.
* Next, cd into mlex_data_clinic
* type `docker-compose up --build` into your terminal

Finally, you can access Data CLinic at:
* Dash app: http://localhost:8072/

Please refer to [HowToGuide](/docs/tasks.md#How To Guide) for further instructions on how
to use this application.

# Model Description
**pytorch_autoencoder:** User-defined autoencoders implemented in [PyTorch](https://pytorch.org).

Further information can be found in [concepts](/docs/concepts.md).

# Copyright
MLExchange Copyright (c) 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.

