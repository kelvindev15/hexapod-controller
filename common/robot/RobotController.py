from abc import ABC, abstractmethod
from typing import Any, List

class RobotController(ABC):
    @abstractmethod
    def goFront(self, distance: float = 1.0) -> None:
        pass

    @abstractmethod
    def goBack(self, distance: float = 1.0) -> None:
        pass

    @abstractmethod
    def rotateRight(self, angle: float = 45.0) -> None:
        pass

    @abstractmethod
    def rotateLeft(self, angle: float = 45.0) -> None:
        pass
    
    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def getCameraImage(self) -> Any:
        pass

    @abstractmethod
    def getLidarImage(self, fov_degrees: int, offset_degrees: int = 0) -> List[float]:
        pass

    @abstractmethod
    def getFrontLidarImage(self) -> List[float]:
        pass
