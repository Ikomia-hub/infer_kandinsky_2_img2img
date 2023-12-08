from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_kandinsky_2_img2img.infer_kandinsky_2_img2img_process import InferKandinskyImg2imgFactory
        return InferKandinskyImg2imgFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_kandinsky_2_img2img.infer_kandinsky_2_img2img_widget import InferKandinskyImg2imgWidgetFactory
        return InferKandinskyImg2imgWidgetFactory()
