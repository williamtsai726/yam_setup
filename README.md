# YAM DAta Collection (json)
The current data collection with camera collected:
    {
        "language_instruction": "testing",
        "left_raw_action": "[0.06937824934720993, -0.00010153520270250738, -0.0036047063767910004, -0.14338549971580505, 0.06944022327661514, 0.010194350965321064, 1.0]",
        "left_delta_action": "[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
        "right_raw_action": "[0.038613419979810715, 0.4039478898048401, -0.001194400480017066, 0.44247910380363464, 0.013759195804595947, -0.042561329901218414, 1.0]",
        "right_delta_action": "[0.0021019999999999373, -0.00035458000000000017, 0.004477700000000001, 0.9999925567007315, 0.0021699664520927406, 0.0009417001963707514, 0.0030481124440036866]",
        "left_joint": "[0.0661860074769205, 0.004386968795300206, 0.056267643244067855, -0.12188143739986579, 0.05321583886472858, 0.00705729762722207, 1.000499097174538]",
        "right_joint": "[0.02117189288166621, 0.2161058976119623, 0.05550469214923304, 0.07991912718394723, -0.0009536888685399703, -0.010490577553978753, 0.9990685638717227]",
        "image_front_rgb": "/home/prior/Desktop/YAM/yam_realtime/yam_realtime/scripts/delta_trajectory/testing/000001/front_rgb/000012.png",
        "image_left_rgb": "/home/prior/Desktop/YAM/yam_realtime/yam_realtime/scripts/delta_trajectory/testing/000001/left_rgb/000012.png",
        "image_right_rgb": "/home/prior/Desktop/YAM/yam_realtime/yam_realtime/scripts/delta_trajectory/testing/000001/right_rgb/000012.png"
    },
the raw actions are also joint that are computed by ik while the joint are obs valued, but since we use raw actions when executing env.step(), use raw actions to train
# YAM Realtime Control Interfaces

YAM Realtime is a modular software stack for realtime control, teleoperation, and policy integration on bi-manual I2RT YAM arms.

It provides extensible pythonic infrastructure for low-latency joint command streaming, agent-based policy control, visualization, and integration with inverse kinematics solvers like [pyroki](https://github.com/chungmin99/pyroki) developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

![yam_realtime](media/yam_realtime.gif)

Shown is a headless-capable web-based real-time visualizer and controller for viewing commanded joint state and actual robot state, built with [Viser](https://viser.studio/main/)

## Installation
Clone the repository and initialize submodules:
```bash
git clone --recurse-submodules https://github.com/uynitsuj/yam_realtime.git
# Or if already cloned without --recurse-submodules, run:
git submodule update --init --recursive
```
Install the main package and I2RT repo for CAN driver interface using uv:
```bash
cd yam_realtime
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install dependencies/i2rt/
```
## Configuration
First configure YAM arms CAN chain according to instructions from the [I2RT repo](https://github.com/i2rt-robotics/i2rt)

## Launch
Then run the launch entrypoint script with an appropriate robot config file:
```bash
python yam_realtime/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
```
## Extending with Custom Agents
To integrate your own controller or policy:

Subclass the base agent interface:
```python
from yam_realtime.agents.agent import Agent

class MyAgent(Agent):
    ...
```
Add your agent to your YAML config so the launcher knows which controller to instantiate.

Examples of agents you might implement:
- Leader arm or VR controller teleoperation
- Learned policy (e.g., Diffusion Policy, ACT, PI0)
- Offline motion-planner + scripted trajectory player

## Linting
If contributing, please use ruff (automatically installed) for linting (https://docs.astral.sh/ruff/tutorial/#getting-started)
```bash
ruff check # lint
ruff check --fix # lint and fix anything fixable
ruff format # code format
```

## Roadmap/Todos

- [ ] Add data logging infrastructure
- [ ] Implement a [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) agent controller
- [ ] Implement a [Physical Intelligence π0](https://www.physicalintelligence.company/blog/pi0) agent controller
