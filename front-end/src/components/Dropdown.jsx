import React from 'react';
import Dropdown from 'react-bootstrap/Dropdown';
function DropdownLayout (){
    return(
        <Dropdown>
            <Dropdown.Toggle variant="success" id="dropdown-basic">
            Dropdown Button
            </Dropdown.Toggle>
    
            <Dropdown.Menu>
                <Dropdown.Item href="#/action-1">meta-llama/Llama-2-7b</Dropdown.Item>
                <Dropdown.Item href="#/action-2">mistralai/Mixtral-8x7B-v0.1</Dropdown.Item>
                <Dropdown.Item href="#/action-3">iocuydi/llama-2-amharic-3784m</Dropdown.Item>
            </Dropdown.Menu>
        </Dropdown>
      
    );
}


export default DropdownLayout;