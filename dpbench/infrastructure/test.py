# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
from typing import Any, Callable, Dict, Sequence, Tuple

from dpbench.infrastructure import Benchmark, Framework
from dpbench.infrastructure import timeout_decorator as tout
from dpbench.infrastructure import utilities as util


class Test(object):
    """A class for testing a framework on a benchmark."""

    def __init__(
        self, bench: Benchmark, frmwrk: Framework, npfrmwrk: Framework = None
    ):
        self.bench = bench
        self.frmwrk = frmwrk
        self.numpy = npfrmwrk

    def _execute(
        self,
        frmwrk: Framework,
        impl: Callable,
        impl_name: str,
        mode: str,
        bdata: Dict[str, Any],
        repeat: int,
    ) -> Tuple[Any, Sequence[float]]:
        report_str = frmwrk.info["full_name"] + " - " + impl_name
        try:
            copy = frmwrk.copy_func()
            setup_str = frmwrk.setup_str(self.bench, impl)
            exec_str = frmwrk.exec_str(self.bench, impl)
        except Exception as e:
            print("Failed to load the {} implementation.".format(report_str))
            print(e)
            return None, None
        ldict = {"__dpb_impl": impl, "__dpb_copy": copy, **bdata}
        if setup_str:
            # Execute the copy of arrays into the "__npb_" prefix arrays
            exec(setup_str, ldict)
        try:
            out, timelist = util.benchmark(
                stmt=exec_str,
                out_text=report_str + " - " + mode,
                repeat=repeat,
                context=ldict,
                output="__dpb_result",
            )
        except Exception as e:
            print("Failed to execute the {} implementation.".format(report_str))
            print(e)
            return None, None
        if out is not None:
            if isinstance(out, (tuple, list)):
                out = list(out)
            else:
                out = [out]
        else:
            out = []
        if "output_args" in self.bench.info.keys():
            out += [
                ldict[a]
                for b in self.bench.info["output_args"]
                for a in frmwrk.args(self.bench)
                if b in a
            ]
        return out, timelist

    def run(
        self, preset: str, validate: bool, repeat: int, timeout: float = 200.0
    ):
        """Tests the framework against the benchmark.
        :param preset: The preset to use for testing (S, M, L, paper).
        :param validate: If true, it validates the output against NumPy.
        :param repeat: The number of repetitions.
        """
        print(
            "***** Testing the {b} implementation using {f} on the {p} dataset *****".format(
                b=self.bench.bname, f=self.frmwrk.info["full_name"], p=preset
            )
        )
        bdata = self.bench.get_data(preset)

        # Run NumPy for validation
        if validate and self.frmwrk.fname != "numpy" and self.numpy:
            np_impl, np_impl_name = self.numpy.implementations(self.bench)[0]
            np_out, _ = self._execute(
                frmwrk=self.numpy,
                impl=np_impl,
                impl_name=np_impl_name,
                mode="validation",
                bdata=bdata,
                repeat=1,
            )
            if not np_out:
                print(
                    "Failed to produce an output for the NumPy implementation "
                    + "of "
                    + self.bench.bname
                    + ". Skipping execution of the benchmark."
                )
                return
        else:
            validate = False
            np_out = None

        # Extra information
        kind = ""
        if "kind" in self.bench.info.keys():
            kind = self.bench.info["kind"]
        domain = ""
        if "domain" in self.bench.info.keys():
            domain = self.bench.info["domain"]
        dwarf = ""
        if "dwarf" in self.bench.info.keys():
            dwarf = self.bench.info["dwarf"]
        version = self.frmwrk.version()

        @tout.exit_after(timeout)
        def first_execution(impl, impl_name):
            return self._execute(
                self.frmwrk, impl, impl_name, "first/validation", context, 1
            )

        bvalues = []
        context = {**bdata, **self.frmwrk.imports()}
        for impl, impl_name in self.frmwrk.implementations(self.bench):
            # First execution
            try:
                frmwrk_out, _ = first_execution(impl, impl_name)
            except KeyboardInterrupt:
                print(
                    'Implementation "{}" timed out.'.format(impl_name),
                    flush=True,
                )
                continue
            except Exception:
                raise

            # Validation
            valid = True
            if validate:
                try:
                    validator_fn = self.frmwrk.validator()
                    valid = validator_fn(np_out, frmwrk_out)

                    if valid:
                        print(
                            "{} - {} - validation: SUCCESS".format(
                                self.frmwrk.info["full_name"], impl_name
                            )
                        )
                    else:
                        print(
                            "{} - {} - validation: FAILURE".format(
                                self.frmwrk.info["full_name"], impl_name
                            )
                        )
                except Exception:
                    print(
                        "Failed to run {} validation.".format(
                            self.frmwrk.info["full_name"]
                        )
                    )

            # Main execution
            _, timelist = self._execute(
                self.frmwrk, impl, impl_name, "median", context, repeat
            )
            if timelist:
                for t in timelist:
                    bvalues.append(
                        dict(details=impl_name, validated=valid, time=t)
                    )

        # create a database connection
        database = r"dpbench.db"
        conn = util.create_connection(database)

        # create tables
        if conn is not None:
            # create results table
            util.create_table(conn, util.sql_create_results_table)
        else:
            print("Error! cannot create the database connection.")

        # Write data
        timestamp = int(time.time())
        for d in bvalues:
            new_d = {
                "timestamp": timestamp,
                "benchmark": self.bench.info["short_name"],
                "kind": kind,
                "domain": domain,
                "dwarf": dwarf,
                "preset": preset,
                "mode": "main",
                "framework": self.frmwrk.info["simple_name"],
                "version": version,
                "details": d["details"],
                "validated": d["validated"],
                "time": d["time"],
            }
            result = tuple(new_d.values())
            # print(result)
            util.create_result(conn, util.sql_insert_into_results_table, result)
