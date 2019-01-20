import numpy as np


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # iteration
        self.iter = 0
        # bad frames
        self.bad_iters = []

    def average_iterations(self, n=10):
        """
        Takes the average of the iterations and stores it in the best
        :return:
        """
        if self.bestx is None:
            self.bestx = np.mean(np.asarray(self.recent_xfitted), axis=0)

        else:
            if len(self.recent_xfitted) > n:
                tempx = self.recent_xfitted[len(self.recent_xfitted)-n:]
                self.bestx = np.mean(np.asarray(tempx), axis=0)

        return self
