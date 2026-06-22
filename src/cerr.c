#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>


static PyObject *
mse(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy;  // Time series
    double *x, *y;      // Access to time series

    PyObject *op, *oj, *oh;     // Point, Jacobian and Hessian, as PyObjects
    PyArrayObject *p, *j, *h;   // Point, Jacobian and Hessian

    // Temporary variables
    double mse;
    double e;
    double ja_inc, jb_inc, jc_inc; // TODO MULTI

    // Error checking
    bool success;

    // New style numpy iterator
    NpyIter *i;                 // The iterator
    PyArrayObject *ops[2];      // Arrays iterated over, used in constructor
    npy_uint32 op_flags[2];     // Data access per array, used in constructor
    NpyIter_IterNextFunc* inext;    // Iterator "next block" function
    npy_intp *istride;    // Stride inside current block
    npy_intp *iinnersize; // Number of items in current block
    npy_intp icount;      // Number of items remaining in current block
    char **idata;       // Char pointer array, from iterator
    char *cx, *cy;      // Char pointers to x and y

    if (!PyArg_ParseTuple(args, "OOOO", &ox, &oy, &op, &oj)) {
        //PyErr_SetString(PyExc_Exception, "Incorrect input arguments.");
        return NULL;
    }

    // Assumptions to be ensured by Python code:
    //  x and y are arrays
    //  same length, dimension 1
    //  float64 type
    // order is C
    if (!PyArray_Check(ox)) {
        PyErr_SetString(PyExc_TypeError, "x should be a numpy array");
        goto fail;
    }
    if (!PyArray_Check(oy)) {
        PyErr_SetString(PyExc_TypeError, "y should be a numpy array");
        goto fail;
    }
    if (!PyArray_Check(op)) {
        PyErr_SetString(PyExc_TypeError, "p should be a numpy array");
        goto fail;
    }
    if (!PyArray_Check(oj)) {
        PyErr_SetString(PyExc_TypeError, "J should be a numpy array");
        goto fail;
    }

    // Cast arrays
    ops[0] = (PyArrayObject*)ox;
    ops[1] = (PyArrayObject*)oy;
    p = (PyArrayObject*)op;
    j = (PyArrayObject*)oj;

    // Check array sizes
    if (PyArray_NDIM(ops[0]) != 1) {
        PyErr_SetString(PyExc_TypeError, "x should be 1-dimensional");
        goto fail;
    }
    if (PyArray_NDIM(ops[1]) != 1) {
        PyErr_SetString(PyExc_TypeError, "y should be 1-dimensional");
        goto fail;
    }
    if (PyArray_SIZE(ops[0]) != PyArray_SIZE(ops[1])) {
        PyErr_SetString(PyExc_TypeError, "x and y should be the same size");
        goto fail;
    }
    if (PyArray_NDIM(p) != 1) {
        PyErr_SetString(PyExc_TypeError, "p should be 1-dimensional");
        goto fail;
    }
    if (PyArray_NDIM(j) != 1) {
        PyErr_SetString(PyExc_TypeError, "J should be 1-dimensional");
        goto fail;
    }
    if (PyArray_SIZE(p) != PyArray_SIZE(j)) {
        PyErr_SetString(PyExc_TypeError, "p and J should be the same size");
        goto fail;
    }

    // TODO: Check p is sensible size
    double ninv = 1.0 / (double)(PyArray_SIZE(ops[0]));
    double ninv2 = 2.0 / (double)(PyArray_SIZE(ops[0]));
    int np = 3;
    double *ar = (double*)PyArray_DATA(p);
    double a = *ar;
    ar++;
    double b = *ar;
    ar++;
    double c = *ar;

    double ja, jb, jc;




    // Create iterator
    op_flags[0] = NPY_ITER_READONLY;
    op_flags[1] = NPY_ITER_READONLY;
    i = NpyIter_MultiNew(2, ops,
        NPY_ITER_EXTERNAL_LOOP | NPY_ITER_ZEROSIZE_OK,
        NPY_KEEPORDER,
        NPY_NO_CASTING,
        op_flags,
        NULL);
    if(i == NULL) {
        // Nothing allocated yet, exception is set by MultiNew, so return
        return NULL;
    }

    // Pointers to iterator functions and data
    inext = NpyIter_GetIterNext(i, NULL);
    idata = NpyIter_GetDataPtrArray(i);
    istride = NpyIter_GetInnerStrideArray(i);
    iinnersize = NpyIter_GetInnerLoopSizePtr(i);

    // Calculate
    mse = 0.0;
    ja = 0.0; jb = 0.0; jc = 0.0;
    do {
        // Data pointers for current block
        cx = idata[0];
        cy = idata[1];

        // Inner loop over current block
        icount = *iinnersize;
        while (icount--) {
            x = (double *)cx;
            y = (double *)cy;

            e = exp(c * *x);
            ja_inc = a - *y + b * e;
            jb_inc = ja_inc * e;
            jc_inc = jb_inc * *x;

            mse += ja_inc * ja_inc;
            ja += ja_inc;
            jb += jb_inc;
            jc += jc_inc;

            // Update data pointers
            cx += istride[0];
            cy += istride[1];
        }

    } while (inext(i));

    mse *= ninv;

    ar = (double*)PyArray_DATA(j);
    *ar = ja * ninv2; ar++;
    *ar = jb * ninv2; ar++;
    *ar = jc * ninv2 * b;

    /* Finished succesfully, free memory and return */
    success = 1;
fail:
    /* Free memory */
    NpyIter_Deallocate(i);

    /* Return */
    if (success) {
        return PyFloat_FromDouble(mse);
    } else {
        return 0;
    }
}

static PyMethodDef cerr_module_methods[] = {
    {"mse",  mse, METH_VARARGS, "Testing an MSE in C."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// Module "execution" slot - second phase on initialisation
static bool initialised = false;
static int
cerr_module_exec(PyObject *m)
{
    if (initialised) {
        PyErr_SetString(PyExc_ImportError, "Cannot initialize expfit._cerr module more than once");
        return -1;
    }
    initialised = true;

    return 0;
}

// List of slot definitions for multi-phase initialization
static PyModuleDef_Slot cerr_module_slots[] = {
    // Not useing Py_mod_create slot
    {Py_mod_exec, cerr_module_exec},
    {0, NULL}
};

// Module definition
static struct PyModuleDef cerr_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_cerr",
    .m_size = 0, // Assume module has no state
    .m_slots = cerr_module_slots,
    .m_methods = cerr_module_methods
};

// Initialisation function
PyMODINIT_FUNC
PyInit__cerr(void)
{
    // Initialise numpy
    // Note: This is a macro that will set an error and return NULL if
    // an error occurs
    import_array();

    return PyModuleDef_Init(&cerr_module);
}
