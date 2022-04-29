import sys
def main():
    args=sys.argv[1:]
    assert len(args) == 1, 'Only 1 Argument accepted'
    if args[0] in ["GoSafeOpt","SafeOpt"]:
        from controller_optimization_impedance_path import experiment
        experiment(args[0])
    elif args[0]=="eic":
        import controller_optimization_impedance_path_eic
    
    else:
        assert False, 'Only GoSafeOpt, SafeOpt or eic accepted'
        
    
if __name__ == "__main__":
    main()
