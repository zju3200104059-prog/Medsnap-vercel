# -*- coding: utf-8 -*-
"""
MedSnap 研究成果管理 — Flask Blueprint 路由
所有 /api/results/* 路由。
"""

import json
import sqlite3
import os
import io
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file

import pandas as pd

# 科室配置（与 app.py / statistics_routes.py 保持一致）
DEPARTMENT_CONFIGS = {
    'cardiology':  {'name': '心内科',   'color': '#dc2626'},
    'neurology':   {'name': '神经内科', 'color': '#7c3aed'},
    'surgery':     {'name': '外科',     'color': '#0891b2'},
    'pediatrics':  {'name': '儿科',     'color': '#f59e0b'},
    'obstetrics':  {'name': '妇产科',   'color': '#ec4899'},
    'emergency':   {'name': '急诊科',   'color': '#ef4444'},
    'general':     {'name': '通用',     'color': '#64748b'},
}


def _resolve_dept_name(dept_id):
    """将 dept_id 映射为科室中文名"""
    return DEPARTMENT_CONFIGS.get(dept_id, {}).get('name', dept_id or '未知')


research_bp = Blueprint('research', __name__)


# ==============================================================================
#  辅助函数
# ==============================================================================

def _get_db():
    """获取数据库连接"""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_medical_data.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ==============================================================================
#  API 路由
# ==============================================================================

@research_bp.route('/api/results', methods=['GET'])
def list_results():
    """列表/搜索研究成果，支持分页、排序和筛选"""
    try:
        keyword = request.args.get('keyword', '').strip()
        dept_id = request.args.get('dept_id', '').strip()
        data_type = request.args.get('data_type', '').strip()
        sort_order = request.args.get('sort', 'desc').strip().lower()
        if sort_order not in ('asc', 'desc'):
            sort_order = 'desc'
        page = max(1, int(request.args.get('page', 1)))
        page_size = min(100, max(1, int(request.args.get('page_size', 20))))

        conn = _get_db()
        c = conn.cursor()

        conditions = []
        params = []

        if keyword:
            conditions.append(
                "(result_id LIKE ? OR title LIKE ? OR summary LIKE ? OR conclusion LIKE ? OR dept_id LIKE ?)"
            )
            kw = f'%{keyword}%'
            params.extend([kw, kw, kw, kw, kw])

        if dept_id:
            conditions.append("dept_id = ?")
            params.append(dept_id)

        if data_type:
            conditions.append("data_type = ?")
            params.append(data_type)

        where_clause = (' WHERE ' + ' AND '.join(conditions)) if conditions else ''

        # 总数
        count_sql = f"SELECT COUNT(*) FROM research_results{where_clause}"
        total = c.execute(count_sql, params).fetchone()[0]

        # 分页查询（支持排序方向）
        offset = (page - 1) * page_size
        order = 'ASC' if sort_order == 'asc' else 'DESC'
        data_sql = f"SELECT * FROM research_results{where_clause} ORDER BY create_time {order} LIMIT ? OFFSET ?"
        rows = c.execute(data_sql, params + [page_size, offset]).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item['dept_name'] = _resolve_dept_name(item.get('dept_id', ''))
            # 解析 JSON 字段
            for field in ['core_metrics', 'notes']:
                val = item.get(field)
                if val:
                    try:
                        item[field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(item)

        conn.close()
        return jsonify({
            "status": "success",
            "data": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@research_bp.route('/api/results', methods=['POST'])
def create_result():
    """创建新研究成果"""
    try:
        data = request.get_json(force=True)
        result_id = str(uuid.uuid4())
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        core_metrics = data.get('core_metrics', '')
        if isinstance(core_metrics, (dict, list)):
            core_metrics = json.dumps(core_metrics, ensure_ascii=False)

        notes = data.get('notes', '[]')
        if isinstance(notes, (list,)):
            notes = json.dumps(notes, ensure_ascii=False)

        conn = _get_db()
        c = conn.cursor()
        c.execute('''INSERT INTO research_results
            (result_id, data_type, dept_id, source_record_id, title, summary,
             core_metrics, conclusion, notes, status, create_time, update_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (result_id,
             data.get('data_type', ''),
             data.get('dept_id', 'general'),
             data.get('source_record_id', ''),
             data.get('title', ''),
             data.get('summary', ''),
             core_metrics,
             data.get('conclusion', ''),
             notes,
             data.get('status', '待复核'),
             now, now))
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "result_id": result_id, "msg": "成果已创建"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@research_bp.route('/api/results/<result_id>', methods=['GET'])
def get_result(result_id):
    """获取研究成果详情"""
    try:
        conn = _get_db()
        c = conn.cursor()
        row = c.execute("SELECT * FROM research_results WHERE result_id = ?", (result_id,)).fetchone()
        conn.close()

        if not row:
            return jsonify({"status": "error", "msg": "成果不存在"})

        item = dict(row)
        item['dept_name'] = _resolve_dept_name(item.get('dept_id', ''))
        for field in ['core_metrics', 'notes']:
            val = item.get(field)
            if val:
                try:
                    item[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass

        return jsonify({"status": "success", "data": item})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@research_bp.route('/api/results/<result_id>', methods=['PUT'])
def update_result(result_id):
    """更新研究成果（主要用于备注管理）"""
    try:
        data = request.get_json(force=True)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = _get_db()
        c = conn.cursor()

        # 检查是否存在
        existing = c.execute("SELECT result_id FROM research_results WHERE result_id = ?", (result_id,)).fetchone()
        if not existing:
            conn.close()
            return jsonify({"status": "error", "msg": "成果不存在"})

        # 构建动态更新
        update_fields = []
        update_values = []
        allowed_fields = ['title', 'summary', 'conclusion', 'notes', 'dept_id', 'data_type', 'status', 'core_metrics']

        for field in allowed_fields:
            if field in data:
                val = data[field]
                if field in ['notes', 'core_metrics'] and isinstance(val, (dict, list)):
                    val = json.dumps(val, ensure_ascii=False)
                update_fields.append(f"{field} = ?")
                update_values.append(val)

        update_fields.append("update_time = ?")
        update_values.append(now)
        update_values.append(result_id)

        sql = f"UPDATE research_results SET {', '.join(update_fields)} WHERE result_id = ?"
        c.execute(sql, update_values)
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "msg": "更新成功"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@research_bp.route('/api/results/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    """删除研究成果"""
    try:
        conn = _get_db()
        c = conn.cursor()
        c.execute("DELETE FROM research_results WHERE result_id = ?", (result_id,))
        deleted = c.rowcount
        conn.commit()
        conn.close()

        if deleted:
            return jsonify({"status": "success", "msg": "已删除"})
        else:
            return jsonify({"status": "error", "msg": "成果不存在"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@research_bp.route('/api/results/export', methods=['POST'])
def export_results():
    """批量导出研究成果为 Excel"""
    try:
        data = request.get_json(force=True)
        result_ids = data.get('result_ids', [])

        if not result_ids:
            return jsonify({"status": "error", "msg": "请选择要导出的成果"})

        conn = _get_db()
        c = conn.cursor()
        placeholders = ','.join(['?' for _ in result_ids])
        rows = c.execute(
            f"SELECT * FROM research_results WHERE result_id IN ({placeholders}) ORDER BY create_time DESC",
            result_ids
        ).fetchall()
        conn.close()

        if not rows:
            return jsonify({"status": "error", "msg": "未找到选中的成果"})

        # 构建 DataFrame
        records = []
        for row in rows:
            item = dict(row)
            item['dept_name'] = _resolve_dept_name(item.get('dept_id', ''))
            # 将 JSON 字段转为可读文本
            for field in ['core_metrics', 'notes']:
                val = item.get(field)
                if val:
                    try:
                        parsed = json.loads(val)
                        item[field] = json.dumps(parsed, ensure_ascii=False, indent=2)
                    except (json.JSONDecodeError, TypeError):
                        pass
            records.append(item)

        df = pd.DataFrame(records)

        # 列名映射
        col_map = {
            'result_id': '成果ID', 'data_type': '数据类型', 'dept_id': '科室ID',
            'dept_name': '科室', 'source_record_id': '来源记录',
            'title': '标题', 'summary': '摘要', 'core_metrics': '核心指标',
            'conclusion': '分析结论', 'notes': '备注', 'status': '状态',
            'create_time': '创建时间', 'update_time': '更新时间'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)

        return send_file(
            buf,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'research_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@research_bp.route('/api/results/<result_id>/export', methods=['GET'])
def export_single_result(result_id):
    """单条导出研究成果为 Excel"""
    try:
        conn = _get_db()
        c = conn.cursor()
        row = c.execute("SELECT * FROM research_results WHERE result_id = ?", (result_id,)).fetchone()
        conn.close()

        if not row:
            return jsonify({"status": "error", "msg": "成果不存在"})

        item = dict(row)
        item['dept_name'] = _resolve_dept_name(item.get('dept_id', ''))
        for field in ['core_metrics', 'notes']:
            val = item.get(field)
            if val:
                try:
                    parsed = json.loads(val)
                    item[field] = json.dumps(parsed, ensure_ascii=False, indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass

        df = pd.DataFrame([item])

        col_map = {
            'result_id': '成果ID', 'data_type': '数据类型', 'dept_id': '科室ID',
            'dept_name': '科室', 'source_record_id': '来源记录',
            'title': '标题', 'summary': '摘要', 'core_metrics': '核心指标',
            'conclusion': '分析结论', 'notes': '备注', 'status': '状态',
            'create_time': '创建时间', 'update_time': '更新时间'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)

        short_id = result_id[:8].upper()
        return send_file(
            buf,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'research_result_{short_id}_{datetime.now().strftime("%Y%m%d")}.xlsx'
        )
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})