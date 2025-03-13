from torch import device, backends, cuda


class _DeviceManager:
    _device = None

    @staticmethod
    def init_device() -> device:
        if _DeviceManager._device is None:
            if backends.mps.is_available():
                _DeviceManager._device = device('mps')  # macOS GPU
            elif cuda.is_available():
                _DeviceManager._device = device('cuda')  # Linux/Windows GPU
            else:
                _DeviceManager._device = device('cpu')
            return _DeviceManager._device
        else:
            print(f"warning: init_device(): {_DeviceManager._device} is already being used.")

    @staticmethod
    def get_device() -> device:
        if _DeviceManager._device:
            return _DeviceManager._device
        else:
            print("warning: get_device(): device has not been initialized.")
            return None


init_device = _DeviceManager.init_device
get_device = _DeviceManager.get_device