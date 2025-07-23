
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'MIN_net':
        from .MIN import MIN
        model = MIN()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
