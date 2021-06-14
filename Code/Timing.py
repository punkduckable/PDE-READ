import time



class Timer_Exception(Exception):
    # Raised if the user does uses a timer object incorrectly.
    pass;



class Timer:
    def __init__(self):
        self.Start_Time : float = None;

    def Start(self) -> None:
        # Record the current time. Note that if the user calls this function
        # multiple times, then we'll will just override the starting time.
        self.Start_Time = time.perf_counter();

    def Stop(self) -> float:
        # Make sure the timer has actually started.
        if (self.Start_Time is None):
            raise Timer_Exception("Timer has not started. Bad");

        # Return the time elapsed since the timer started
        return time.perf_counter() - self.Start_Time;
