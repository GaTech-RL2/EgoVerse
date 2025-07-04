import numpy as np
import matplotlib.pyplot as plt

def save_image(image, path):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

class TemporalAgg:
    def __init__(self):
        self.recent_actions = []
    
    def add_action(self, action):
        """
            actions: (100, 7) tensor
        """
        self.recent_actions.append(action)
        if len(self.recent_actions) > 4:
            del self.recent_actions[0]

    def smoothed_action(self):
        """
            returns smooth action (100, 7)
        """
        mask = []
        count = 0

        shifted_actions = []
        # breakpoint()

        for ac in self.recent_actions[::-1]:
            basic_mask = np.zeros(100)
            basic_mask[:100-count] = 1
            mask.append(basic_mask)
            shifted_ac = ac[count:]
            shifted_ac = np.concatenate([shifted_ac, np.zeros((count, 7))], axis=0)
            shifted_actions.append(shifted_ac)
            count += 25

        mask = mask[::-1]
        mask = ~(np.array(mask).astype(bool))
        recent_actions = shifted_actions[::-1]
        recent_actions = np.array(recent_actions)
        # breakpoint()
        mask = np.repeat(mask[:, :, None], 7, axis=2)
        smoothed_action = np.ma.array(recent_actions, mask=mask).mean(axis=0)

        # PLOT_JOINT = 0
        # for i in range(recent_actions.shape[0]):
        #     plt.plot(recent_actions[i, :, PLOT_JOINT], label=f"index{i}")
        # plt.plot(smoothed_action[:, PLOT_JOINT], label="smooth")
        # plt.legend()
        # plt.savefig("smoothing.png")
        # plt.close()
        # breakpoint()

        return smoothed_action