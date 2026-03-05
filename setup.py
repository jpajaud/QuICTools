from setuptools import setup, find_packages

setup(
    name="QuICTools",
    version="0.0.0",
    description="Useful functions for quantum numerics in QuIC B",
    author="Jon Pajaud",
    author_email="jpajaud2@gmail.com",
    packages=find_packages(),  # ['quictools','quictools.spins','quictools.models.pspin','quictools.quantum','quictools.constants']
    package_data={"quictools.grape": ["data/*"], "quictools": ["data/*"]},
    zip_safe=False,
)
