.. Mininumpy documentation master file, created by
   sphinx-quickstart on Wed Oct 29 16:06:24 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Build Your Own MiniNumPy Library
================================

Homework of the Programming course M1 International Track in EE - Université Évry Paris-Saclay. 

Authored by Long Quang NGUYEN.

Objective
*****************

The goal of this project is to design and implement a simplified version of NumPy, called *MiniNumPy*, to understand how numerical computing and linear algebra libraries are built from scratch. Students will implement core data structures, array operations, and linear algebra routines.

By the end, students should:


Understand how arrays are represented in memory.

• Implement basic array manipulation (reshape, transpose, slicing).

• Write elementwise and matrix operations.

• Explore algorithms for linear algebra (determinant, inverse, eigenvalues).

Project Tasks
*********************

Part 1: Core Array Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


• Implement a class `Array <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array>`_ that wraps a Python ``list`` (or nested ``list``).
• Store attributes: `.data <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.data>`_, `.shape <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.shape>`_, `.ndim <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.ndim>`_, `.size <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.size>`_.
• Add methods:
• `reshape(new_shape) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.reshape>`_
• `transpose() <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.transpose>`_
• `__str__ <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.__str__>`_ for pretty printing

Part 2: Array Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
• Implement helper functions:
• `array(list_or_nested_list) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.array>`_
• `zeros(shape) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.zeros>`_
• `ones(shape) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.ones>`_
• `eye(n) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.eye>`_
• `arange(start, stop, step) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.arange>`_
• `linspace(start, stop, num) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.linspace>`_

Part 3: Elementwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
• Overload Python operators (``+,-,/,*,/,**``).

.. toctree::
   :maxdepth: 4
   :caption: API reference

   modules

