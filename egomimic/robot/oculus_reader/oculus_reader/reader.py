from oculus_reader.FPS_counter import FPSCounter
from oculus_reader.buttons_parser import parse_buttons
import numpy as np
import threading
import time
import os
from ppadb.client import Client as AdbClient
import sys


def eprint(*args, **kwargs):
    RED = "\033[1;31m"
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)


class OculusReader:
    def __init__(
        self,
        ip_address=None,
        port=5555,
        APK_name="com.rail.oculus.teleop",
        print_FPS=False,
        run=True,
    ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        self._lock = threading.Lock()
        self.tag = "wE9ryARX"

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        if self.print_FPS:
            self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self):
        self.running = True
        self.device.shell(
            'am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER'
        )
        self.thread = threading.Thread(
            target=self.device.shell, args=("logcat -T 0", self.read_logcat_by_line)
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()

    # ---------- Device selection helpers ----------
    def _adb_client(self):
        host = os.environ.get("ANDROID_ADB_SERVER_HOST", "127.0.0.1")
        port = int(os.environ.get("ADB_SERVER_PORT", "5037"))
        return AdbClient(host=host, port=port)

    def _pick_by_serial(self, client, serial):
        if not serial:
            return None
        try:
            dev = client.device(serial)
            if dev:
                return dev
        except RuntimeError:
            pass
        return None

    def _pick_quest_by_properties(self, client):
        try:
            for d in client.devices():
                # Prefer model name containing "quest"
                props = {}
                try:
                    props = d.get_properties() or {}
                except Exception:
                    props = {}
                model = (props.get("ro.product.model", "") or "").lower()
                manufacturer = (props.get("ro.product.manufacturer", "") or "").lower()
                brand = (props.get("ro.product.brand", "") or "").lower()
                if "quest" in model or "quest" in brand or "oculus" in manufacturer:
                    return d
        except RuntimeError:
            os.system("adb devices")
            for d in client.devices():
                try:
                    props = d.get_properties() or {}
                except Exception:
                    props = {}
                model = (props.get("ro.product.model", "") or "").lower()
                manufacturer = (props.get("ro.product.manufacturer", "") or "").lower()
                brand = (props.get("ro.product.brand", "") or "").lower()
                if "quest" in model or "quest" in brand or "oculus" in manufacturer:
                    return d
        return None

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system("adb devices")
            client.remote_connect(self.ip_address, self.port)
        device = client.device(f"{self.ip_address}:{self.port}")

        if device is None:
            if retry == 1:
                os.system(f"adb tcpip {self.port}")
            if retry == 2:
                eprint(
                    "Make sure the device is running and available at the provided ip_address."
                )
                eprint("Currently provided IP address:", self.ip_address)
                eprint("Run `adb shell ip route` to verify the IP address.")
                sys.exit(1)
            else:
                return self.get_network_device(client=client, retry=retry + 1)
        return device

    def get_usb_device(self, client):
        """
        Select the Quest explicitly:
        1) Respect QUEST_SERIAL or ANDROID_SERIAL if set.
        2) Otherwise scan all USB devices and pick one whose properties indicate Quest.
        3) As a last resort, fail with a clear message instead of picking the first device.
        """
        quest_serial = os.environ.get("QUEST_SERIAL") or os.environ.get(
            "ANDROID_SERIAL"
        )
        dev = self._pick_by_serial(client, quest_serial)
        if dev:
            return dev

        dev = self._pick_quest_by_properties(client)
        if dev:
            return dev

        # 3) No unambiguous Quest device found -> help the user
        eprint("No Quest device selected/found.")
        eprint("Tip: set QUEST_SERIAL (or ANDROID_SERIAL) to your Quest’s serial.")
        eprint("Example:")
        eprint("  export QUEST_SERIAL=1WMHHXXXXXXX")
        eprint("  export ANDROID_SERIAL=$QUEST_SERIAL")
        eprint("Then re-run your launch.")
        # Show devices to aid debugging
        try:
            lst = client.devices()
            if not lst:
                eprint("adb sees no devices. Check cable/USB debugging.")
            else:
                eprint("adb devices detected:")
                for d in lst:
                    try:
                        props = d.get_properties() or {}
                    except Exception:
                        props = {}
                    eprint(f"- {d.serial} :: {props.get('ro.product.model', '')}")
        except Exception:
            os.system("adb devices")
        sys.exit(1)

    def get_device(self):
        client = self._adb_client()
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    # ---------- Install / Uninstall ----------
    def install(self, APK_path=None, verbose=True, reinstall=False):
        """
        Skips reinstall if:
        - app already present, or
        - OCULUS_SKIP_INSTALL=1
        """
        try:
            skip_env = os.environ.get("OCULUS_SKIP_INSTALL", "0") not in (
                "0",
                "",
                "false",
                "False",
            )
            already = self.device.is_installed(self.APK_name)
            if skip_env:
                if verbose:
                    print("Skipping APK install due to OCULUS_SKIP_INSTALL.")
                return
            if not already or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "APK",
                        "teleop-debug.apk",
                    )
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                now_installed = self.device.is_installed(self.APK_name)
                if now_installed and success:
                    print("APK installed successfully.")
                else:
                    eprint("APK install failed.")
            elif verbose:
                print("APK is already installed.")
        except RuntimeError:
            eprint("Device is visible but could not be accessed.")
            eprint(
                "Run `adb devices` to verify that the device is visible and accessible."
            )
            eprint(
                'If you see "no permissions" next to the device serial, allow USB debugging on the Quest.'
            )
            sys.exit(1)

    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print("APK uninstall finished.")
                    print(
                        'Please verify the app disappeared from the list as described in "UNINSTALL.md".'
                    )
                    print(
                        "For potential issues, see https://github.com/Swind/pure-python-adb/issues/71."
                    )
                else:
                    eprint("APK uninstall failed")
            elif verbose:
                print("APK is not installed.")
        except RuntimeError:
            eprint("Device is visible but could not be accessed.")
            eprint(
                "Run `adb devices` to verify that the device is visible and accessible."
            )
            eprint(
                'If you see "no permissions" next to the device serial, allow USB debugging on the Quest.'
            )
            sys.exit(1)

    # ---------- Data path ----------
    @staticmethod
    def process_data(string):
        try:
            transforms_string, buttons_string = string.split("&")
        except ValueError:
            return None, None
        split_transform_strings = transforms_string.split("|")
        transforms = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4, 4))
            pair = pair_string.split(":")
            if len(pair) != 2:
                continue
            left_right_char = pair[0]  # 'r' or 'l'
            transform_string = pair[1]
            values = transform_string.split(" ")
            c = 0
            r = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[r][c] = float(value)
                c += 1
                if c >= 4:
                    c = 0
                    r += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def extract_data(self, line):
        output = ""
        if self.tag in line:
            try:
                output += line.split(self.tag + ": ")[1]
            except ValueError:
                pass
        return output

    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()


def main():
    oculus_reader = OculusReader()
    while True:
        time.sleep(0.3)
        print(oculus_reader.get_transformations_and_buttons())


if __name__ == "__main__":
    main()
