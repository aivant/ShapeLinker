import os
import subprocess
import io

class ShapeAlignment():

    def calculate_alignment_score(self, query_file, smiles, model_path, query_type, alignment_repo_path, 
                                alignment_env, num_conformers, poses_folder, es_weight, step, 
                                mode = 'linkinvent', correct_flipping = False):
        if mode == 'linkinvent':
            filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'link_invent' , 'linker_shape_scoring_submit.py')
        else:
            raise ValueError('Mode not supported')
        command = self._create_command(
            query_file, 
            smiles, 
            model_path, 
            query_type, 
            filepath, 
            alignment_env, 
            num_conformers, 
            poses_folder, 
            step, 
            alignment_repo_path, 
            es_weight, 
            correct_flipping)
        current_scores = self._print_cmd_retrieve_results(command, len(smiles))
        current_scores = [float(score) for score in current_scores]
        
        return current_scores
    
    def _create_command(
        self, 
        query_file, 
        smiles, 
        model_path, 
        query_type, 
        script_path, 
        alignment_env, 
        num_conformers, 
        poses_folder, 
        step, 
        alignment_repo_path, 
        es_weight, 
        correct_flipping):
        concat_smiles = '"' + ';'.join(smiles) + '"'
        query_file = '"' + query_file + '"'
        command = ' '.join([alignment_env,
                            script_path,
                            '--model_path', model_path,
                            '--query_file', query_file,
                            '--query_type', query_type,
                            '--smiles_cmd', concat_smiles,
                            '--num_conformers', str(num_conformers),
                            '--output_folder', poses_folder,
                            '--step', str(step),
                            '--alignment_repo_path', alignment_repo_path,
                            '--es_weight', str(es_weight),
                            '--correct_flipping' if correct_flipping else '',
                            ])

        return command

    def _print_cmd_retrieve_results(self, command, data_size: int):
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=True) as proc:
            wrapped_proc_in = io.TextIOWrapper(proc.stdin, 'utf-8')
            wrapped_proc_out = io.TextIOWrapper(proc.stdout, 'utf-8')
            result = [self._parse_result(wrapped_proc_out.readline()) for i in range(data_size)]
            wrapped_proc_in.close()
            wrapped_proc_out.close()
            proc.wait()
            proc.terminate()
        return result
    
    def _parse_result(self, result):
        return str(result).strip()