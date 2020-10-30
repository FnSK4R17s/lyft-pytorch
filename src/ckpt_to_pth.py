from Lightning import LyftNet
import torch
import config


input = os.path.join(conifg.INPUT, 'latest_model.ckpt')
output = os.path.join(conifg.OUTPUT, 'saved_model.pth')


lit_model = LyftNet(hparams)
lit_model = lit_model.load_from_checkpoint(checkpoint_path=input)

torch.save(lit_model.model.state_dict(), output)

print("Done !")