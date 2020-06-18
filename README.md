# 2048-api
A 2048 game api based on  supervised learning (imitation learning)
# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`get_data.py`](game2048/get_data.py): to generate the data.
    * [`module.py`](game2048/module.py): the core model 'high_net' class.
    * [`para/`](game2048/para): to save the train parameters.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
    
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
* [`train_high.py`](game2048/train_high.py): to train the model and save parameters.
* [`use_on_cloud.ipynb`](game2048/use_on_cloud.ipynb): to get data and train the model and save parameters on the Cloud services.

# Requirements
* [`get_data.py`](game2048/get_data.py):should be run on windows.If it is used on linux, the[`expectimax/`](game2048/expectimax) should be changed.
File name can be changed to save data in different files.
* The data files are saved on the Cloud services.


# To get data

```bash
python get_data.py
```

# To train model(on gpu)

```bash
python train_high.py
```

# To evaluate the agent based on trained model

```bash
python evaluate.py > evaluate_result.log
```

# LICENSE
The code is under Apache-2.0 License.

