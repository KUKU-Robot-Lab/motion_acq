from glob import glob

from setuptools import setup

package_name = "motion_acq_hand"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="divingyoon",
    maintainer_email="ukwh5681@gmail.com",
    description="Nova 2 -> RH56F1 hand teleop nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "hand_node = motion_acq_hand.hand_node:main",
            "fake_glove = motion_acq_hand.fake_glove_node:main",
            "fake_rh56f1 = motion_acq_hand.fake_rh56f1_node:main",
            "calibrate = motion_acq_hand.calibrate:main",
        ],
    },
)
