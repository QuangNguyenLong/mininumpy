.. |ci-badge| image:: https://github.com/QuangNguyenLong/mininumpy/actions/workflows/ci.yml/badge.svg
   :alt: CI Status

Build Your Own MiniNumPy Library |ci-badge|
===========================================

Homework of the Programming course M1 International Track in EE - Université Évry Paris-Saclay.

Authored by Long Quang NGUYEN.

Objective
*****************

The goal of this project is to design and implement a simplified version of NumPy, called *MiniNumPy*, to understand how numerical computing and linear algebra libraries are built from scratch. Students will implement core data structures, array operations, and linear algebra routines.

By the end, students should:


Understand how arrays are represented in memory.

- Implement basic array manipulation (reshape, transpose, slicing).

- Write elementwise and matrix operations.

- Explore algorithms for linear algebra (determinant, inverse, eigenvalues).

Project Tasks
*********************

Part 1: Core Array Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Implement a class `Array <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array>`_ that wraps a Python ``list`` (or nested ``list``).
- Store attributes: `.data <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.data>`_, `.shape <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.shape>`_, `.ndim <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.ndim>`_, `.size <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.size>`_.
- Add methods:
   - `reshape(new_shape) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.reshape>`_
   - `transpose() <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.transpose>`_
   - `__str__ <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.__str__>`_ for pretty printing

Part 2: Array Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Implement helper functions:
   - `array(list_or_nested_list) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.array>`_
   - `zeros(shape) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.zeros>`_
   - `ones(shape) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.ones>`_
   - `eye(n) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.eye>`_
   - `arange(start, stop, step) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.arange>`_
   - `linspace(start, stop, num) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.linspace>`_

Part 3: Elementwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Overload Python operators (`__add__ <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.__add__>`_, `__sub__ <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.__sub__>`_, `__mul__ <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.__mul__>`_, `__truefiv__ <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.__truefiv__>`_, `__pow__ <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.__pow__>`_).
- Implement elementwise functions:
   - `exp <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.__exp__>`_, `log <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.__log__>`_, `sqrt <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.__sqrt__>`_, `abs <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.__abs__>`_.
- Implement reductions:
   - `sum <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.sum>`_, `mean <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.mean>`_, `min <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.min>`_, `max <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.max>`_, `argmin <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.argmin>`_, `argmax <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.array.Array.argmax>`_.

Part 4: Linear Algebra Module (`minilinalg <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#module-mininumpy.minilinalg>`_)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Implement matrix/vector operations:
   - `dot(a, b) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.minilinalg.dot>`_  - dot product / matrix multiply.
   - `matmul(a, b) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.minilinalg.matmul>`_  - general matrix multiplication (@ operator).
   - `norm(a) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.minilinalg.norm>`_ - vector/matrix norm.

- Implement basic factorizations/solvers:
   - `det(a) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.minilinalg.det>`_  - determinant (via recursion or LU).
   - `inv(a) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.minilinalg.inv>`_  - matrix inverse.
   - `eig(a) <https://quangnguyenlong.github.io/mininumpy/mininumpy.html#mininumpy.minilinalg.eig>`_  - eigenvalues and eigenvectors (bonus).
   
Part 5: Applications (Mini-Projects)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Students must demonstrate their library with **practical applications**:
   1. `Image manipulation <https://github.com/QuangNguyenLong/mininumpy/blob/main/ex1.py>`_ (grayscale filter or rotation using matrices).
   2. `2D transformation <https://github.com/QuangNguyenLong/mininumpy/blob/main/ex2.py>`_: scale, rotate, and shear a set of points.

Deliverables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   1. Source code of MiniNumPy (mininumpy/array.py, mininumpy/linalg.py).
   2. A **report (5-10 pages)** explaining design choices and algorithms.
   3. A `demo notebook <https://github.com/QuangNguyenLong/mininumpy/blob/main/sample.ipynb>`_ showing use cases and comparisons with real NumPy.