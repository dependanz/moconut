# Model Construction Utils (moconut)

We borrow the conventions of functional patch construction from MaxMSP/Puredata/etc.. and present a python library of tools to create complex models:
1. Each module has a number of inlets, taking in objects.
2. Each module does something with those objects.
3. Each module has a number of outlets, each outputs objects.

