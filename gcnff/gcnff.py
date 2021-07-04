import sys
from gcnff.printsave   import Logger
from gcnff.showplot    import showfig
from gcnff.helpinfo    import showhelp
from gcnff.get_graph   import get_graph
from gcnff.model_test  import model_test
from gcnff.divide_set  import divide_set
from gcnff.get_potential import get_potential
from gcnff.AutomaticWeightedLoss import AutomaticWeightedLoss
from gcnff.divide_train  import divide_fix_train,divide_free_train
from gcnff.direct_train  import direct_fix_train,direct_free_train

def main():
    if len(sys.argv) < 2:
        showhelp()
        exit(-1)
    action = sys.argv[1]
    if action == '-help' or action == '-h':
        showhelp()
        exit(0)
    if action not in ['get_graph', 'divide_set', 'divide_fix_train', 'divide_free_train', 'model_test', 'get_potential','showfig','direct_fix_train','direct_free_train']:
        showhelp()
        exit(-1)
    sys.stdout=Logger('gcnff.log')
    config_file = sys.argv[2]
    if action == 'get_graph':
        get_graph(config_file)

    if action == 'divide_set':
        divide_set(config_file)

    if action == 'divide_fix_train':
        divide_fix_train(config_file)

    if action == 'divide_free_train':
        divide_free_train(config_file)

    if action == 'direct_fix_train':
        direct_fix_train(config_file)
        
    if action == 'direct_free_train':
        direct_free_train(config_file)
        
    if action == 'model_test':
        model_test(config_file)

    if action == 'get_potential':
        get_potential(config_file)
        
    if action == 'showfig':
        showfig(config_file)
        
if __name__ == '__main__':
    main()
