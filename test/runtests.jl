using ManyExpertDecisionTrees
using Test
using DataFrames
using Statistics
using SoleLogics

import DecisionTree: build_tree
import ManyExpertDecisionTrees: SigmoidHyperParameters, AbstractMembershipFunction

@testset "ManyExpertDecisionTrees.jl" begin
    
    @testset "subdivide" begin
        X = [1:10 11:20]
        
        parts = ManyExpertDecisionTrees.subdivide(3, X)
        @test length(parts) == 3
        @test size(parts[1], 1) == 3
        @test size(parts[2], 1) == 3
        @test size(parts[3], 1) == 4  
        
        parts = ManyExpertDecisionTrees.subdivide(1, X)
        @test length(parts) == 1
        @test size(parts[1], 1) == 10
    end
    
   
    @testset "Leaf" begin
        leaf = ManyExpertDecisionTrees.MEDTLeaf(1);
        @test leaf.label == 1
        @test length(leaf) == 1
        @test depth(leaf) == 0
    end
    
    @testset "Node" begin
        leaf1 = ManyExpertDecisionTrees.MEDTLeaf(0)
        leaf2 = ManyExpertDecisionTrees.MEDTLeaf(1)
        
        node = ManyExpertDecisionTrees.MEDTNode(
            0.5, 
            1,
            leaf1, 
            leaf2,
            [SigmoidMF(0.0,1.0, Ref(SigmoidHyperParameters()))], 
            [SigmoidMF(0.0, 1.0, Ref(SigmoidHyperParameters()))]
        )
        
        @test node.featval == 0.5
        @test node.featid == 1
        @test length(node) == 2
        @test depth(node) == 1
    end
    
    @testset "manify" begin
        X = [1.0 1.0; 2.0 1.0; 3.0 2.0; 4.0 2.0; 5.0 3.0; 6.0 3.0]
        y1 = [0, 0, 1, 1, 1, 1]
        
        dt1 = build_tree(y1, X)
        
        medt = manify(dt1, X, SigmoidMF, SigmoidMF)
        
        @test medt isa ManyExpertDecisionTree
        @test medt.nfeats == 2  
        @test length(medt.mftypes) == 2
        @test length(medt) >= 1
        @test depth(medt) >= 0
    end
    
    @testset "apply" begin
        using SoleLogics.ManyValuedLogics
        
        X = [1.0 1.0; 2.0 1.0; 3.0 2.0; 4.0 2.0; 5.0 3.0; 6.0 3.0]
        y1 = [0, 0, 1, 1, 1, 1]
        y2 = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        
        dt1 = build_tree(y1, X)

        dt2 = build_tree(y2, X)

        medt = manify(dt1, X, SigmoidMF, SigmoidMF)
        
        experts = (GodelLogic, GodelLogic)
        
        instance1 = [1.5, 1.0]
        result1 = apply(medt, experts, instance1)
        
        @test result1 isa Vector
        @test !isempty(result1)
        
        instance2 = [5.0, 3.0]
        result2 = apply(medt, experts, instance2)
        
        @test result2 isa Vector
        @test !isempty(result2)
        
        wrong_dim_instance = [1.0]  
        @test_throws ErrorException apply(medt, experts, wrong_dim_instance)
        
        experts_wrong = (GodelLogic,)
        @test_throws ErrorException apply(medt, experts_wrong, instance1)
        
        medt3 = manify(dt1, X, SigmoidMF, SigmoidMF, SigmoidMF)
        experts = (GodelLogic, ProductLogic, LukasiewiczLogic)
        
        result3 = apply(medt3, experts, instance1)
        @test result3 isa Vector
        @test !isempty(result3)

        medt4 = fuzzify(dt2, X, SigmoidMF)
        expert = GodelLogic
        result4 = apply(medt4, expert, instance1; depth=2)
        @test !isempty(result4)
    end
    
    # TODO: Re-enable when addexperts! is reimplemented
    # @testset "addexperts!" begin
    #     X = DataFrame(
    #         x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    #         x2 = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    #     )
    #     y = [0, 0, 1, 1, 1, 1]
    #     
    #     dt = build_tree(y, Matrix(X))
    #     X_matrix = Matrix(X)
    #     medt = manify(dt, X_matrix, FL.GaussianMF, FL.GaussianMF)
    #     
    #     initial_expert_count = length(medt.mftypes)
    #     initial_length = length(medt)
    #     
    #     addexperts!(medt, X_matrix, FL.GaussianMF)
    #     
    #     @test length(medt.mftypes) == initial_expert_count + 1
    #     @test length(medt) == initial_length  # Tree structure unchanged
    #     @test medt.mftypes[end] == FL.GaussianMF{Float64}
    #     
    #     root = medt.root
    #     if root isa ManyExpertDecisionTrees.MEDTNode
    #         @test length(root.mfleft) == initial_expert_count + 1
    #         @test length(root.mfright) == initial_expert_count + 1
    #     end
    #     
    #     addexperts!(medt, X_matrix, FL.GaussianMF, FL.GaussianMF)
    #     
    #     @test length(medt.mftypes) == initial_expert_count + 3
    #     @test medt.mftypes[end-1] == FL.GaussianMF{Float64}
    #     @test medt.mftypes[end] == FL.GaussianMF{Float64}
    #     
    #     if medt.root isa ManyExpertDecisionTrees.MEDTNode
    #         @test length(medt.root.mfleft) == initial_expert_count + 3
    #         @test length(medt.root.mfright) == initial_expert_count + 3
    #     end
    # end     
end
