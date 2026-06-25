from setuptools import setup

package_name = "kf_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        # Ament package marker.
        ("share/ament_index/resource_index/packages", ["resource/kf_bringup"]),
        (f"share/{package_name}", ["package.xml"]),
        # Launch and config files installed into share/.
        (f"share/{package_name}/launch", ["launch/eskf_kitti.launch.py"]),
        (f"share/{package_name}/config", ["config/eskf_kitti.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Derek Wang",
    maintainer_email="info@prandtldynamics.com",
    description="Launch files, config, and KITTI replay node for the kf_eskf stack.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "kitti_replay = kf_bringup.kitti_replay:main",
        ],
    },
)
