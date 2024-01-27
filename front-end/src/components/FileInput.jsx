import React from 'react';
import 'bootstrap/dist/css/bootstrap.css'
function FileInput (){
    return(
        <div>
            <div class="input-group mb-3">
                <input type="file" class="form-control" id="inputGroupFile02"/>
                <label class="input-group-text" for="inputGroupFile02">Upload</label>
            </div>
        </div>
      
    );
}


export default FileInput;