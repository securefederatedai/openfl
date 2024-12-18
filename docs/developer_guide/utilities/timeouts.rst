.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

*******************************************************
OpenFL Component Timeouts
*******************************************************

.. _comp_timeout_overview:

Overview
========

This feature allows decorating any arbitrary synchronous and/or asynchronous functions using :code:`@fedtiming(timeout=<seconds>)`. 
The decorated functions is then monitored and gets terminated right after the execution time exceeds the user specified or default timeout value.

`openfl.utilities.fed_timer.py`

.. note::
     
    The `fedtiming` class, `SyncAsyncTaskDecoFactory` factory class, custom synchronous and asynchronous execution of decorated function is in-place. The end to end implementation of OpenFL Component timeouts feature is still in beta mode and would undergo design and implementation changes before the complete feature is made available. Appreciate any feedbacks or issues.


.. _comp_timeout_design:

Class Diagram
===========================

An overview of this workflow is shown below.

.. figure:: /images/timeout_design.png

.. class:: center
 Overview of the component timeout class diagram



.. _comp_timeout_flow_of_execution:

Flow of execution
===================

#. [Step A] Decorate any sync or async function :code:`@fedtiming(timeout=<seconds>)` to monitor its execution time and terminate after `timeout=<seconds>` value.


      .. code-block:: shell

        @fedtiming(timeout=5)
        def some_sync_function():
            pass

      | This decorated function execution gets terminated after `5 seconds`.

      .. code-block:: shell

        @fedtiming(timeout=10)
        async def some_async_function():
            await some_long_running_operation()

      | This decorated function execution gets terminated after `10 seconds`.

#. [Step B] Concrete `fedtiming` class:

    **During Compile time:** Decorated functions are evaluated like below.
       
      **Synchronous Example:**

      .. code-block:: shell

        some_sync_function = fedtiming(timeout=5)(some_sync_function)

        then 

        some_sync_function() *is equivalent to* sync_wrapper().

      inside the sync_wrapper: the decorated function `some_sync_function` and `timeout` variables are stored as a closure variable.
        
      **Aynchronous Example:**

      .. code-block:: shell

        some_async_function = fedtiming(timeout=5)(some_async_function)

        then 

        some_async_function() *is equivalent to* async_wrapper().

      inside the async_wrapper: the decorated function `some_async_function` and `timeout` variables are stored as a closure variable.
        
    
#. [Step C] `SyncAsyncTaskDecoFactory` class 

    `fedtiming(some_sync_function)` internally calls the parent class `SyncAsyncTaskDecoFactory` :code:`__call__(some_sync_function)` method.
    
    The :code:`__call__()` method immediately returns either the `sync_wrapper` or `async_wrapper` depending on whether the sync or async method was decorated.
      

    **During Runtime:**
       
     The prepared `some_sync_function` or `some_async_function` when called internally with its respective parameters.

     .. code-block:: shell

      some_sync_function(*args, **kwargs) -> sync_wrapper(*args, **kwargs)
      some_async_function(*args, **kwargs) -> async_wrapper(*args, **kwargs)


#. [Step D] `PrepareTask` class
    
    Delegates the decorated sync or async function to be executed synchronously or asynchronously using `CustomThread` or `asyncio`.
    
    Contains the defination for the function `sync_execute` and `async_execute`.

#. [Step E] Execution of delegated methods:

    The delegated function is executed synchronously or asynchronously and the result is returned back in the call chain.
    The final output from the `thread` or `asyncio` task is returned as a result of a decorated function execution.

    In this `CustomThread` or `asyncio.wait_for()` execution, the timeout is enforced which terminates the running function after a set period of time and an exception is called that tracebacks to the caller.

.. _comp_timeout_upcoming_feature:

Upcoming Changes
===================

**Above design reflects current implementation.**

Upcoming changes include:

 1. Dynamic timeout parameters updates for all decorated functions during runtime. Removal of `timeout` parameter `@fedtiming(timeout=<?>)`.

 2. Add a callback parameter that defines a post timeout teardown logic and a way gracefully terminate executing function.
