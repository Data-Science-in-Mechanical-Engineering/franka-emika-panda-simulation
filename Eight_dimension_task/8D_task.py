import sys
def main():
    args=sys.argv[1:]
    assert len(args) == 1, 'Only 1 Argument accepted'
    if args[0]=="GoSafeOpt":
        import controller_optimization_lowdim_impedance_gosafeopt
    elif args[0]=="SafeOpt":
        import controller_optimization_lowdim_impedance_safeopt
    
    else:
        assert False, 'Only GoSafeOpt or SafeOpt accepted'
        
    
if __name__ == "__main__":
    main()
